from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from .llm_ner import LLMNER
from .nltk_ner import NLTKNER


@dataclass
class NLTKLLMNERResult:
    span: str
    start: int
    end: int
    go_id: str
    go_name: str
    namespace: str
    score: float
    source: str                 # nltk / llm / nltk+llm
    source_scores: Dict[str, float] = field(default_factory=dict)


class NLTKLLMNER:
    """
    NLTK + LLM 级联联合识别：
      1) 先用 NLTKNER 获取高精度字面命中
      2) 再用 LLMNER 补充 NLTK 漏掉的实体
      3) 按 go_id 聚合，输出融合结果
    """

    def __init__(
        self,
        obo_path: str,
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "dummy",
        model: str = "deepseek-r1:14b",
        llm_threshold: float = 0.55,
    ) -> None:
        self.llm_threshold = llm_threshold

        self._nltk_ner = NLTKNER(obo_path=obo_path)
        self._llm_ner = LLMNER(
            obo_path=obo_path,
            api_base=api_base,
            api_key=api_key,
            model=model,
            normalizer="dict",
            conservative=True,
        )

    def recognize(self, text: str, lang: str = "auto") -> List[NLTKLLMNERResult]:
        merged: Dict[str, NLTKLLMNERResult] = {}

        nltk_hits = self._nltk_ner.recognize(text)
        for r in nltk_hits:
            merged[r.go_id] = NLTKLLMNERResult(
                span=r.span,
                start=r.start,
                end=r.end,
                go_id=r.go_id,
                go_name=r.go_name,
                namespace=r.namespace,
                score=r.score,
                source="nltk",
                source_scores={"nltk": r.score},
            )

        llm_hits = self._llm_ner.recognize(text, lang=lang)
        for r in llm_hits:
            if not r.go_id:
                continue
            if r.confidence < self.llm_threshold:
                continue

            m = re.search(re.escape(r.original_span), text, flags=re.IGNORECASE)
            if m:
                span = text[m.start():m.end()]
                start, end = m.start(), m.end()
            else:
                span = r.original_span
                start, end = -1, -1

            prev = merged.get(r.go_id)
            if prev is None:
                merged[r.go_id] = NLTKLLMNERResult(
                    span=span,
                    start=start,
                    end=end,
                    go_id=r.go_id,
                    go_name=r.go_name or "",
                    namespace=r.namespace or "",
                    score=r.confidence,
                    source="llm",
                    source_scores={"llm": r.confidence},
                )
            else:
                prev.source = "nltk+llm"
                prev.source_scores["llm"] = r.confidence
                prev.score = max(prev.score, r.confidence)

        results = list(merged.values())
        results.sort(key=lambda x: (x.start if x.start >= 0 else 10**9, -x.score))
        return results

    def format_results(self, results: List[NLTKLLMNERResult]) -> str:
        if not results:
            return "未识别到任何 GO 术语。"

        lines = [f"NLTK+LLM 共识别 {len(results)} 个实体：\n"]
        for r in results:
            pos = f"[{r.start},{r.end}]" if r.start >= 0 else "[未定位]"
            lines.append(
                f"  [{r.source}] {pos} \"{r.span}\"\n"
                f"      -> {r.go_id} | {r.go_name}\n"
                f"      namespace: {r.namespace} | score: {r.score:.3f} | source_scores: {r.source_scores}\n"
            )
        return "\n".join(lines)
