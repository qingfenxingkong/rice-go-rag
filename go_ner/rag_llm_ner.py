from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from .dict_ner import DictNER
from .llm_ner import LLMNERResult
from .obo_parser import GOEntry, parse_obo
from .vector_ner import VectorNER


@dataclass
class RetrievedGOCandidate:
    go_id: str
    go_name: str
    namespace: str
    definition: str
    score: float
    source: str


class RAGLLMNER:
    """检索增强的 LLM GO 实体识别。"""

    _GENERIC_BLACKLIST = {
        "development", "process", "response", "regulation", "activity",
        "function", "organization", "pathway", "mechanism", "system",
        "signaling", "expression", "biosynthesis", "metabolism", "growth",
        "integrity", "homeostasis", "cascade", "network", "machinery",
    }

    _SYSTEM_PROMPT = """You are a biomedical ontology grounding expert for Gene Ontology (GO).
Read the text and choose only valid GO terms from the provided retrieved candidate list.
Return a JSON array. Each item must be an object with keys: go_id, span, confidence.
Rules:
- Only choose GO terms clearly supported by the text.
- Only choose from the provided candidate list.
- If no candidate is clearly supported, return [].
Example output:
[{"go_id":"GO:0016301","span":"kinase activity","confidence":0.92}]
"""

    _SYSTEM_PROMPT_ZH = """你是一名 Gene Ontology (GO) 术语判别专家。
请阅读输入文本，并且仅从给定候选 GO 列表中选择被文本明确支持的术语。
返回 JSON 数组。每个元素必须包含：go_id, span, confidence。
规则：
- 只能从给定候选列表中选择。
- 如果没有明确支持的候选，返回 []。
示例输出：
[{"go_id":"GO:0016301","span":"kinase activity","confidence":0.92}]
"""

    def __init__(self, obo_path: str, model_path: str, api_base: str = "http://localhost:11434/v1", api_key: str = "dummy", model: str = "deepseek-r1:14b", index_path: Optional[str] = None, metadata_path: Optional[str] = None, candidate_top_k: int = 12, vector_threshold: float = 0.70, conservative: bool = True) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._candidate_top_k = candidate_top_k
        self._vector_threshold = vector_threshold
        self._conservative = conservative
        print("[RAGLLMNER] 加载 OBO 文件...")
        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)
        self._dict_ner = DictNER(obo_path=obo_path, use_fuzzy=True)
        self._vector_ner = VectorNER(obo_path=obo_path, model_path=model_path, index_path=index_path, metadata_path=metadata_path, top_k=max(candidate_top_k, 5))

    def _detect_lang(self, text: str, lang: str) -> str:
        if lang != "auto":
            return lang
        zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return "zh" if zh_count / max(len(text), 1) > 0.1 else "en"

    def _collect_candidates(self, text: str) -> List[RetrievedGOCandidate]:
        merged: Dict[str, RetrievedGOCandidate] = {}
        for r in self._dict_ner.recognize(text):
            if r.span.strip().lower() in self._GENERIC_BLACKLIST:
                continue
            entry = self._entries.get(r.go_id)
            if entry:
                merged[r.go_id] = RetrievedGOCandidate(r.go_id, entry.name, entry.namespace, entry.definition, r.score, f"dict_{r.match_type}")
        for hit in self._vector_ner.search(text, top_k=self._candidate_top_k):
            if hit.similarity < self._vector_threshold:
                continue
            entry = self._entries.get(hit.go_id)
            if not entry:
                continue
            cand = RetrievedGOCandidate(hit.go_id, entry.name, entry.namespace, entry.definition, hit.similarity, "vector")
            prev = merged.get(hit.go_id)
            if prev is None or cand.score > prev.score:
                merged[hit.go_id] = cand
        return sorted(merged.values(), key=lambda x: x.score, reverse=True)[:self._candidate_top_k]

    def _build_user_prompt(self, text: str, candidates: List[RetrievedGOCandidate], lang: str) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            definition = (c.definition or "").replace("\n", " ").strip()
            if len(definition) > 160:
                definition = definition[:160] + "..."
            lines.append(f"{i}. {c.go_id} | {c.go_name} | {c.namespace} | source={c.source} | score={c.score:.3f} | def={definition}")
        block = "\n".join(lines)
        if lang == "zh":
            return f"文本：\n{text}\n\n候选GO列表：\n{block}\n\n请仅从候选列表中选择被文本支持的GO，返回JSON数组。"
        return f"Text:\n{text}\n\nCandidate GO list:\n{block}\n\nSelect only GO terms supported by the text and return a JSON array."

    def _call_llm_select(self, text: str, candidates: List[RetrievedGOCandidate], lang: str) -> List[dict]:
        if not candidates:
            return []
        payload = {"model": self._model, "messages": [{"role": "system", "content": self._SYSTEM_PROMPT_ZH if lang == "zh" else self._SYSTEM_PROMPT}, {"role": "user", "content": self._build_user_prompt(text, candidates, lang)}], "temperature": 0.0 if self._conservative else 0.1, "stream": False}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(f"{self._api_base}/chat/completions", headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            content = re.sub(r'<think>.*?</think>', '', resp.json()["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
        except Exception as ex:
            print(f"[RAGLLMNER] LLM 判别失败: {ex}")
        return []

    def recognize(self, text: str, lang: str = "auto") -> List[LLMNERResult]:
        lang = self._detect_lang(text, lang)
        print("[RAGLLMNER] 检索候选 GO...")
        candidates = self._collect_candidates(text)
        print(f"[RAGLLMNER] 检索到 {len(candidates)} 个候选")
        judged = self._call_llm_select(text, candidates, lang)
        print(f"[RAGLLMNER] LLM 选出 {len(judged)} 个候选")
        candidate_map = {c.go_id: c for c in candidates}
        results: List[LLMNERResult] = []
        seen = set()
        for item in judged:
            go_id = str(item.get("go_id", "")).strip().upper()
            if not go_id or go_id in seen or go_id not in candidate_map:
                continue
            seen.add(go_id)
            entry = self._entries.get(go_id)
            if not entry:
                continue
            try:
                confidence = float(item.get("confidence", candidate_map[go_id].score))
            except Exception:
                confidence = candidate_map[go_id].score
            span = str(item.get("span", entry.name)).strip() or entry.name
            results.append(LLMNERResult(original_span=span, go_id=entry.go_id, go_name=entry.name, namespace=entry.namespace, definition=entry.definition, match_type=f"rag_{candidate_map[go_id].source}", confidence=max(0.0, min(1.0, confidence))))
        return results

    def format_results(self, results: List[LLMNERResult]) -> str:
        if not results:
            return "RAG-LLM 未识别到任何 GO 相关实体。"
        lines = [f"RAG-LLM 共识别 {len(results)} 个实体：\n"]
        for r in results:
            lines.append(f"  原文: \"{r.original_span}\"\n  标准化: {r.go_id} | {r.go_name}\n  命名空间: {r.namespace}\n  方式: {r.match_type} | 置信度: {r.confidence:.3f}\n")
        return "\n".join(lines)
