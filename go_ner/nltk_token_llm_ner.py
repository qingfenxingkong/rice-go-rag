from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from nltk.util import ngrams

from .dict_ner import DictNER
from .obo_parser import GOEntry, parse_obo


@dataclass
class NLTKTokenLLMNERResult:
    original_span: str
    go_id: Optional[str]
    go_name: Optional[str]
    namespace: Optional[str]
    definition: Optional[str]
    match_type: str
    confidence: float


class NLTKTokenLLMNER:
    """
    基于 NLTK 候选短语引导的 LLM GO 实体识别：
      1) 用 NLTK 风格 token + ngram 生成候选短语
      2) 将原文 + 候选短语列表交给 LLM
      3) LLM 只从候选中选择可能的 GO 实体
      4) 用 DictNER 做标准化
    """

    _TOKEN_RE = re.compile(r"[\w\-/]+")
    _GENERIC_BLACKLIST = {
        "development", "process", "response", "regulation", "activity",
        "function", "organization", "pathway", "mechanism", "system",
        "signaling", "expression", "biosynthesis", "metabolism", "growth",
        "integrity", "homeostasis", "cascade", "network", "machinery",
    }
    _STOPWORDS = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "that", "this", "which", "it", "its", "into", "under",
    }

    _SYSTEM_PROMPT = """You are a biomedical GO entity recognition expert.
You are given:
1. the original text
2. a candidate phrase list extracted by NLP

Your task:
- Select ONLY phrases from the candidate list that are likely to refer to Gene Ontology related entities
- Be conservative
- Do not invent new phrases
- Return a JSON array of selected phrases only

If none are valid, return [].
"""

    _SYSTEM_PROMPT_ZH = """你是一名 Gene Ontology (GO) 实体识别专家。
给你两部分内容：
1. 原始文本
2. 由 NLP 方法抽取出的候选短语列表

你的任务：
- 只能从候选短语列表中选择可能属于 GO 相关实体的短语
- 不要自行生成新短语
- 尽量保守
- 返回 JSON 数组（字符串数组）

如果没有合适实体，返回 []。
"""

    def __init__(
        self,
        obo_path: str,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "dummy",
        model: str = "deepseek-r1:14b",
        max_ngram: int = 5,
        llm_threshold: float = 0.55,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self.max_ngram = max_ngram
        self.llm_threshold = llm_threshold

        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)
        self._dict_ner = DictNER(obo_path=obo_path, use_fuzzy=True)

    def _detect_lang(self, text: str, lang: str) -> str:
        if lang != "auto":
            return lang
        zh_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        return "zh" if zh_count / max(len(text), 1) > 0.1 else "en"

    def _token_spans(self, text: str) -> List[Tuple[str, int, int]]:
        spans: List[Tuple[str, int, int]] = []
        for m in self._TOKEN_RE.finditer(text):
            spans.append((m.group(0).lower(), m.start(), m.end()))
        return spans

    def _generate_candidates(self, text: str) -> List[str]:
        tokens = self._token_spans(text)
        token_texts = [t[0] for t in tokens]

        candidates = set()

        for n in range(self.max_ngram, 0, -1):
            for gram in ngrams(token_texts, n):
                words = list(gram)

                while words and words[0] in self._STOPWORDS:
                    words.pop(0)
                while words and words[-1] in self._STOPWORDS:
                    words.pop()

                if not words:
                    continue

                phrase = " ".join(words).strip()

                if len(phrase) < 3:
                    continue
                if phrase in self._GENERIC_BLACKLIST:
                    continue
                if len(phrase.split()) == 1 and phrase in self._STOPWORDS:
                    continue

                candidates.add(phrase)

        ordered = sorted(candidates, key=lambda x: (-len(x.split()), -len(x), x))
        return ordered[:80]

    def _call_llm_select(self, text: str, candidates: List[str], lang: str) -> List[str]:
        if not candidates:
            return []

        system_prompt = self._SYSTEM_PROMPT_ZH if lang == "zh" else self._SYSTEM_PROMPT
        candidate_block = "\n".join(f"- {c}" for c in candidates)

        user_prompt = (
            f"文本：\n{text}\n\n候选短语：\n{candidate_block}\n\n请仅返回候选中属于 GO 实体的短语 JSON 数组。"
            if lang == "zh"
            else f"Text:\n{text}\n\nCandidate phrases:\n{candidate_block}\n\nReturn only a JSON array of selected GO-related phrases."
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
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
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            m = re.search(r"\[.*?\]", content, re.DOTALL)
            if m:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    selected = [str(x).strip() for x in arr if str(x).strip()]
                    selected_set = {c.lower() for c in candidates}
                    return [x for x in selected if x.lower() in selected_set]
        except Exception as ex:
            print(f"[NLTKTokenLLMNER] LLM 调用失败: {ex}")

        return []

    def recognize(self, text: str, lang: str = "auto") -> List[NLTKTokenLLMNERResult]:
        lang = self._detect_lang(text, lang)

        candidates = self._generate_candidates(text)
        selected = self._call_llm_select(text, candidates, lang)

        results: List[NLTKTokenLLMNERResult] = []
        for phrase in selected:
            entry = self._dict_ner.normalize(phrase)
            if not entry:
                results.append(NLTKTokenLLMNERResult(
                    original_span=phrase,
                    go_id=None,
                    go_name=None,
                    namespace=None,
                    definition=None,
                    match_type="none",
                    confidence=0.0,
                ))
                continue

            match_type = "dict_exact" if phrase.lower() == entry.name.lower() else "dict_fuzzy"
            confidence = 0.9 if match_type == "dict_exact" else 0.65

            if confidence < self.llm_threshold:
                continue

            results.append(NLTKTokenLLMNERResult(
                original_span=phrase,
                go_id=entry.go_id,
                go_name=entry.name,
                namespace=entry.namespace,
                definition=entry.definition,
                match_type=match_type,
                confidence=confidence,
            ))

        dedup: Dict[str, NLTKTokenLLMNERResult] = {}
        for r in results:
            if not r.go_id:
                continue
            prev = dedup.get(r.go_id)
            if prev is None or r.confidence > prev.confidence:
                dedup[r.go_id] = r

        return list(dedup.values())

    def format_results(self, results: List[NLTKTokenLLMNERResult]) -> str:
        if not results:
            return "NLTK-token-LLM 未识别到任何 GO 相关实体。"

        lines = [f"NLTK-token-LLM 共识别 {len(results)} 个实体：\n"]
        for r in results:
            lines.append(
                f"  原文短语: \"{r.original_span}\"\n"
                f"  标准化: {r.go_id} | {r.go_name}\n"
                f"  命名空间: {r.namespace}\n"
                f"  匹配方式: {r.match_type} | 置信度: {r.confidence:.3f}\n"
            )
        return "\n".join(lines)
