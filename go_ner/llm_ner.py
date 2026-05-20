"""
go_ner/llm_ner.py
方案三：基于大模型（DeepSeek/Ollama）的 GO 术语 NER + 概念标准化

策略：
  1. 用 LLM 从文本中抽取生物学实体（NER）
  2. 用 DictNER 或 VectorNER 将抽取的实体标准化为 GO ID
  3. 最智能，能处理复杂语境、中英文混合文本
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .obo_parser import GOEntry, parse_obo


@dataclass
class LLMNERResult:
    """LLM NER 结果。"""
    original_span: str          # LLM 抽取的原始实体文本
    go_id: Optional[str]        # 标准化后的 GO ID（可能为 None）
    go_name: Optional[str]      # 标准名称
    namespace: Optional[str]    # 命名空间
    definition: Optional[str]   # 定义
    match_type: str             # 标准化方式: dict_exact / dict_fuzzy / vector / none
    confidence: float           # 置信度


class LLMNER:
    """
    基于大模型的 GO 术语 NER + 概念标准化。

    Args:
        obo_path:       go.obo 文件路径
        api_base:       LLM API 地址（OpenAI 兼容）
        api_key:        API Key
        model:          模型名称
        normalizer:     用于标准化的后端，'dict' 或 'vector'
        model_path:     向量模型路径（normalizer='vector' 时需要）
        index_path:     FAISS 索引路径（normalizer='vector' 时可选）
        metadata_path:  元数据路径（normalizer='vector' 时可选）
    """

    # 过于宽泛的词（单独出现时不是有效GO术语）
    _GENERIC_BLACKLIST = {
        "development", "process", "response", "regulation", "activity",
        "function", "organization", "pathway", "mechanism", "system",
        "signaling", "expression", "biosynthesis", "metabolism", "growth",
        "integrity", "homeostasis", "cascade", "network", "machinery",
    }

    _SYSTEM_PROMPT = """You are a biomedical named entity recognition expert specializing in Gene Ontology (GO).
Given a text, extract ONLY specific Gene Ontology terms that are explicitly mentioned.
Be conservative: if unsure, do NOT include the entity.
If no clear GO terms are present, return an empty array [].
Return a JSON array of extracted entities. Each entity should be a string.
Only return the JSON array, no explanation.

Rules:
- Only include terms that map directly to GO biological processes, molecular functions, or cellular components
- Do NOT include vague terms like "development", "response", "process" alone
- Do NOT include agronomic terms (yield, grain, tillering, milling, etc.)
- If no GO terms found, return []

Example output:
["mitochondrion inheritance", "DNA repair", "kinase activity", "nucleus"]
"""

    _SYSTEM_PROMPT_ZH = """你是一名专注于基因本体论（Gene Ontology, GO）的生物医学命名实体识别专家。
给定一段文本，请保守地提取其中明确涉及的生物过程（biological process）、分子功能（molecular function）
和细胞组成（cellular component）实体。

重要要求：
1. 将每个实体翻译为标准的英文 GO 术语名称
2. 以 JSON 数组格式返回（字符串数组），不要添加任何解释
3. 优先使用 Gene Ontology 官方术语名
4. 如果文本中没有明确的 GO 相关实体，返回空数组 []
5. 不要提取农业类术语（产量、粒重、品种、灌溉等）
6. 不要单独提取 "development", "response", "process" 等过于宽泛的词

示例输入：水稻线粒体遗传与细胞分裂密切相关，激酶活性发挥重要作用。
示例输出：["mitochondrion inheritance", "cell division", "kinase activity"]

示例输入：不同施氮量对产量和分蘖数有影响。
示例输出：[]
"""

    def __init__(
        self,
        obo_path: str,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "dummy",
        model: str = "deepseek-r1:14b",
        normalizer: str = "dict",
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        conservative: bool = True,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._conservative = conservative

        print("[LLMNER] 加载 OBO 文件...")
        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)

        # 初始化标准化后端
        if normalizer == "vector" and model_path:
            print("[LLMNER] 使用向量标准化后端...")
            from .vector_ner import VectorNER
            self._normalizer_backend = "vector"
            self._vector_ner = VectorNER(
                obo_path=obo_path,
                model_path=model_path,
                index_path=index_path,
                metadata_path=metadata_path,
            )
        else:
            print("[LLMNER] 使用字典标准化后端...")
            from .dict_ner import DictNER
            self._normalizer_backend = "dict"
            self._dict_ner = DictNER(obo_path=obo_path, use_fuzzy=True)

    def _call_llm(self, text: str, lang: str = "auto") -> List[str]:
        """调用 LLM 抽取生物学实体，返回实体字符串列表。"""
        # 自动判断语言
        if lang == "auto":
            zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            lang = "zh" if zh_count / max(len(text), 1) > 0.1 else "en"

        system_prompt = self._SYSTEM_PROMPT_ZH if lang == "zh" else self._SYSTEM_PROMPT

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"文本：\n{text}" if lang == "zh" else f"Text:\n{text}"},
            ],
            "temperature": 0.0 if self._conservative else 0.1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                f"{self._api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # 去除 think 标签（DeepSeek-R1）
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # 解析 JSON 数组
            m = re.search(r'\[.*?\]', content, re.DOTALL)
            if m:
                entities = json.loads(m.group(0))
                if isinstance(entities, list):
                    return [str(e) for e in entities if e]
        except Exception as ex:
            print(f"[LLMNER] LLM 调用失败: {ex}")

        return []

    def _normalize_entity(self, entity: str) -> Tuple[Optional[GOEntry], str, float]:
        """将实体字符串标准化为 GOEntry，返回 (entry, match_type, confidence)。"""
        # 过滤单个泛化词（黑名单）
        if entity.strip().lower() in self._GENERIC_BLACKLIST:
            return None, "none", 0.0

        if self._normalizer_backend == "vector":
            result = self._vector_ner.normalize(entity)
            if result and result.similarity >= 0.75:  # 提高向量阈值
                entry = self._entries.get(result.go_id)
                return entry, "vector", result.similarity
            return None, "none", 0.0
        else:
            entry = self._dict_ner.normalize(entity)
            if entry:
                # 精确匹配置信度高，模糊匹配需额外检查
                if entity.lower() == entry.name.lower():
                    return entry, "dict_exact", 1.0
                # 模糊匹配：要求实体与GO名称有足够重叠（至少共享主要词）
                entity_words = set(entity.lower().split())
                go_words = set(entry.name.lower().split())
                overlap = len(entity_words & go_words) / max(len(go_words), 1)
                if overlap >= 0.5:  # 至少50%的GO术语词在实体中出现
                    return entry, "dict_fuzzy", 0.6 + 0.3 * overlap
                return None, "none", 0.0
            return None, "none", 0.0

    def recognize(self, text: str, lang: str = "auto") -> List[LLMNERResult]:
        """
        用 LLM 从文本中抽取 GO 相关实体，再标准化为 GO 术语。

        Args:
            text: 输入文本
            lang: 语言 'zh'/'en'/'auto'

        Returns:
            LLMNERResult 列表
        """
        print(f"[LLMNER] 调用 LLM 抽取实体...")
        entities = self._call_llm(text, lang=lang)
        print(f"[LLMNER] LLM 抽取到 {len(entities)} 个实体: {entities}")

        results = []
        for entity in entities:
            entry, match_type, confidence = self._normalize_entity(entity)
            results.append(LLMNERResult(
                original_span=entity,
                go_id=entry.go_id if entry else None,
                go_name=entry.name if entry else None,
                namespace=entry.namespace if entry else None,
                definition=entry.definition if entry else None,
                match_type=match_type,
                confidence=confidence,
            ))
        return results

    def format_results(self, results: List[LLMNERResult]) -> str:
        """格式化识别结果。"""
        if not results:
            return "LLM 未抽取到任何 GO 相关实体。"
        lines = [f"LLM 共抽取 {len(results)} 个实体：\n"]
        for r in results:
            if r.go_id:
                lines.append(
                    f"  原文: \"{r.original_span}\"\n"
                    f"  标准化: {r.go_id} | {r.go_name}\n"
                    f"  命名空间: {r.namespace}\n"
                    f"  方式: {r.match_type} | 置信度: {r.confidence:.3f}\n"
                    f"  定义: {(r.definition or '')[:120]}...\n"
                )
            else:
                lines.append(
                    f"  原文: \"{r.original_span}\"\n"
                    f"  标准化: [未找到匹配的 GO 术语]\n"
                )
        return "\n".join(lines)
