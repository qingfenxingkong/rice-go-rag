"""
go_ner/ensemble_ner.py
生产可用混合级联：DictNER + VectorNER + LLMNER

设计目标：
  - 保留 DictNER 的高精度
  - 用 VectorNER 提升语义召回
  - 用 LLMNER 处理复杂表达
  - 用加权融合 + 一致性增强，降低误报
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .dict_ner import DictNER


@dataclass
class EnsembleNERResult:
    """混合识别的最终结果。"""
    span: str
    start: int
    end: int
    go_id: str
    go_name: str
    namespace: str
    match_type: str          # exact/synonym/fuzzy/go_id/vector/llm
    source: str              # dict / vector / llm / dict+vector / ...
    score: float             # 融合后总分
    source_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class _Candidate:
    """内部候选缓存（按 go_id 聚合）。"""
    go_id: str
    go_name: str
    namespace: str
    span: str
    start: int
    end: int
    match_type: str
    source_scores: Dict[str, float] = field(default_factory=dict)


class EnsembleNER:
    """
    生产可用级联混合 GO NER。

    流程：
      1) DictNER：高精度先行
      2) VectorNER：补召回（仅对未充分覆盖句子）
      3) LLMNER：最后兜底
      4) 融合：按 source 权重计算总分 + 一致性加分
    """

    def __init__(
        self,
        obo_path: str,
        use_dict: bool = True,
        use_vector: bool = True,
        use_llm: bool = True,
        # DictNER
        dict_fuzzy_threshold: float = 0.85,
        dict_use_fuzzy: bool = True,
        # VectorNER
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        vector_top_k: int = 3,
        vector_threshold: float = 0.60,
        # LLMNER
        llm_api_base: str = "http://localhost:11434/v1",
        llm_api_key: str = "dummy",
        llm_model: str = "deepseek-r1:14b",
        llm_threshold: float = 0.50,
        # 融合参数（生产默认）
        weight_dict: float = 0.60,
        weight_vector: float = 0.25,
        weight_llm: float = 0.15,
        consensus_bonus: float = 0.05,
        final_threshold: float = 0.70,
        min_consensus: int = 1,
    ) -> None:
        self.use_dict = use_dict
        self.use_vector = use_vector and bool(model_path)
        self.use_llm = use_llm

        self.vector_threshold = vector_threshold
        self.vector_top_k = vector_top_k
        self.llm_threshold = llm_threshold

        self.weight_dict = weight_dict
        self.weight_vector = weight_vector
        self.weight_llm = weight_llm
        self.consensus_bonus = consensus_bonus
        self.final_threshold = final_threshold
        self.min_consensus = min_consensus

        self._dict_ner: Optional[DictNER] = None
        self._vector_ner = None
        self._llm_ner = None

        if self.use_dict:
            self._dict_ner = DictNER(
                obo_path=obo_path,
                fuzzy_threshold=dict_fuzzy_threshold,
                use_fuzzy=dict_use_fuzzy,
            )

        if self.use_vector:
            from .vector_ner import VectorNER
            self._vector_ner = VectorNER(
                obo_path=obo_path,
                model_path=model_path or "",
                index_path=index_path,
                metadata_path=metadata_path,
                top_k=vector_top_k,
            )

        if self.use_llm:
            from .llm_ner import LLMNER
            # 生产模式：LLM 抽取 + 字典标准化最稳
            self._llm_ner = LLMNER(
                obo_path=obo_path,
                api_base=llm_api_base,
                api_key=llm_api_key,
                model=llm_model,
                normalizer="dict",
            )

    @staticmethod
    def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        return max(0, e - s)

    @staticmethod
    def _sentence_spans(text: str) -> List[Tuple[str, int, int]]:
        spans: List[Tuple[str, int, int]] = []
        for m in re.finditer(r"[^.;\n]+", text):
            seg = m.group(0)
            trimmed = seg.strip()
            if len(trimmed) < 6:
                continue
            left = seg.find(trimmed)
            start = m.start() + left
            end = start + len(trimmed)
            spans.append((trimmed, start, end))
        return spans

    @staticmethod
    def _best_span(a: _Candidate, b_span: str, b_start: int, b_end: int, prefer_existing: bool = True) -> Tuple[str, int, int]:
        """简单策略：优先已有定位；若已有未定位(-1)，使用新定位。"""
        if prefer_existing and a.start >= 0:
            return a.span, a.start, a.end
        if b_start >= 0:
            return b_span, b_start, b_end
        return a.span, a.start, a.end

    def _fused_score(self, source_scores: Dict[str, float]) -> float:
        parts: List[Tuple[float, float]] = []
        if "dict" in source_scores:
            parts.append((self.weight_dict, source_scores["dict"]))
        if "vector" in source_scores:
            parts.append((self.weight_vector, source_scores["vector"]))
        if "llm" in source_scores:
            parts.append((self.weight_llm, source_scores["llm"]))

        if not parts:
            return 0.0

        wsum = sum(w for w, _ in parts)
        base = sum(w * s for w, s in parts) / max(wsum, 1e-8)

        # 一致性加分：多源命中更可信
        consensus = len(source_scores)
        bonus = max(consensus - 1, 0) * self.consensus_bonus

        return min(1.0, base + bonus)

    def recognize(self, text: str) -> List[EnsembleNERResult]:
        """执行级联识别并输出融合结果。"""
        candidates: Dict[str, _Candidate] = {}
        covered_ranges: List[Tuple[int, int]] = []

        # 第1层：DictNER（高精度）
        if self._dict_ner is not None:
            dict_hits = self._dict_ner.recognize(text)
            for r in dict_hits:
                cand = candidates.get(r.go_id)
                if cand is None:
                    candidates[r.go_id] = _Candidate(
                        go_id=r.go_id,
                        go_name=r.go_name,
                        namespace=r.namespace,
                        span=r.span,
                        start=r.start,
                        end=r.end,
                        match_type=r.match_type,
                        source_scores={"dict": r.score},
                    )
                else:
                    cand.source_scores["dict"] = max(cand.source_scores.get("dict", 0.0), r.score)
                    cand.span, cand.start, cand.end = self._best_span(cand, r.span, r.start, r.end)
                    if cand.match_type != "exact" and r.match_type == "exact":
                        cand.match_type = "exact"
                covered_ranges.append((r.start, r.end))

        # 第2层：VectorNER（补召回）
        if self._vector_ner is not None:
            for sent, s_start, s_end in self._sentence_spans(text):
                sent_len = max(s_end - s_start, 1)
                overlap_len = sum(self._overlap((s_start, s_end), rg) for rg in covered_ranges)
                if overlap_len / sent_len > 0.60:
                    continue

                hits = self._vector_ner.search(sent, top_k=self.vector_top_k)
                if not hits:
                    continue
                best = hits[0]
                if best.similarity < self.vector_threshold:
                    continue

                cand = candidates.get(best.go_id)
                if cand is None:
                    candidates[best.go_id] = _Candidate(
                        go_id=best.go_id,
                        go_name=best.go_name,
                        namespace=best.namespace,
                        span=sent,
                        start=s_start,
                        end=s_end,
                        match_type="vector",
                        source_scores={"vector": best.similarity},
                    )
                else:
                    cand.source_scores["vector"] = max(cand.source_scores.get("vector", 0.0), best.similarity)
                    # 若 dict 没定位，使用 vector 句子定位
                    cand.span, cand.start, cand.end = self._best_span(cand, sent, s_start, s_end)

        # 第3层：LLMNER（兜底）
        if self._llm_ner is not None:
            llm_hits = self._llm_ner.recognize(text, lang="auto")
            for h in llm_hits:
                if not h.go_id:
                    continue
                if h.confidence < self.llm_threshold:
                    continue

                m = re.search(re.escape(h.original_span), text, flags=re.IGNORECASE)
                if m:
                    span, start, end = text[m.start():m.end()], m.start(), m.end()
                else:
                    span, start, end = h.original_span, -1, -1

                cand = candidates.get(h.go_id)
                if cand is None:
                    candidates[h.go_id] = _Candidate(
                        go_id=h.go_id,
                        go_name=h.go_name or "",
                        namespace=h.namespace or "",
                        span=span,
                        start=start,
                        end=end,
                        match_type="llm",
                        source_scores={"llm": h.confidence},
                    )
                else:
                    cand.source_scores["llm"] = max(cand.source_scores.get("llm", 0.0), h.confidence)
                    cand.span, cand.start, cand.end = self._best_span(cand, span, start, end)

        # 融合 + 过滤
        final: List[EnsembleNERResult] = []
        for cand in candidates.values():
            fused = self._fused_score(cand.source_scores)
            consensus = len(cand.source_scores)

            # 放行规则（保守模式）：
            # 1) 强 dict 证据：精确/同义词/GO-ID 匹配 且 分数>=0.95
            # 2) 或达到融合分阈值 且 一致性满足（至少两个子模型同意）
            strong_dict = (
                "dict" in cand.source_scores
                and cand.match_type in {"exact", "go_id", "synonym"}
                and cand.source_scores.get("dict", 0.0) >= 0.95
            )

            if not strong_dict:
                if fused < self.final_threshold:
                    continue
                if consensus < self.min_consensus:
                    continue

            source = "+".join(sorted(cand.source_scores.keys()))
            final.append(EnsembleNERResult(
                span=cand.span,
                start=cand.start,
                end=cand.end,
                go_id=cand.go_id,
                go_name=cand.go_name,
                namespace=cand.namespace,
                match_type=cand.match_type,
                source=source,
                score=round(fused, 3),
                source_scores={k: round(v, 3) for k, v in cand.source_scores.items()},
            ))

        final.sort(key=lambda r: (r.start if r.start >= 0 else 10**9, -r.score))
        return final

    def format_results(self, text: str, results: List[EnsembleNERResult]) -> str:
        """格式化输出。"""
        if not results:
            return "未识别到任何 GO 术语。"

        lines = [f"文本：{text[:100]}{'...' if len(text) > 100 else ''}\n"]
        lines.append(f"共识别到 {len(results)} 个 GO 术语（生产级混合级联）：\n")
        for r in results:
            pos = f"[{r.start},{r.end}]" if r.start >= 0 else "[未定位]"
            lines.append(
                f"  [{r.source:14s}/{r.match_type:7s}] {pos} \"{r.span}\"\n"
                f"             -> {r.go_id} | {r.go_name}\n"
                f"             命名空间: {r.namespace} | 融合分: {r.score:.3f} | 分源: {r.source_scores}\n"
            )
        return "\n".join(lines)
