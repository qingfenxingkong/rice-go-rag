from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .go_hierarchy import GOHierarchy


@dataclass
class MergeCandidate:
    """归并前的候选 GO。"""
    go_id: str
    go_name: str
    namespace: str
    score: float
    source: str = "unknown"
    span: str = ""
    match_type: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MergedGO:
    """归并后的最终 GO。"""
    go_id: str
    go_name: str
    namespace: str
    score: float
    kept_from: List[MergeCandidate] = field(default_factory=list)
    dropped_go_ids: List[str] = field(default_factory=list)
    reason: str = ""


class GOMerger:
    """
    基于 GO DAG 的候选术语归并器（初版）。

    设计目标：
      1. 同 namespace 下消除祖先/后代冗余
      2. 同语义簇内只保留一个主概念
      3. 控制每句输出数量，优先保留主概念

    当前版本采用“轻量可解释”策略：
      - 层级关系：使用 GOHierarchy.descendants 判断祖先/后代
      - 语义簇：使用可手工维护的少量概念簇
      - 打分：文本贴合度 + 原模型分数 + 深度奖励
    """

    DEFAULT_LIMITS = {
        "biological_process": 2,
        "molecular_function": 1,
        "cellular_component": 1,
    }

    # 可逐步扩充的语义簇（name 级别）
    DEFAULT_CLUSTERS = {
        "photosynthesis": {
            "photosynthesis",
            "carbon fixation",
            "photosynthetic acclimation",
            "light harvesting",
            "light reaction",
            "photosystem ii assembly",
        },
        "dna_repair": {
            "dna repair",
            "double-strand break repair",
            "genome maintenance",
            "repair factor recruitment",
        },
        "transmembrane_transport": {
            "transmembrane transport",
            "potassium ion transport",
            "ion transport",
            "metabolite transport",
            "transporter activity",
        },
        "signal_transduction": {
            "signal transduction",
            "signaling cascade",
            "mapk cascade",
            "kinase signaling",
            "receptor signaling",
        },
        "translation": {
            "translation",
            "translational initiation",
            "translational elongation",
            "ribosome biogenesis",
            "ribosomal assembly",
        },
    }

    def __init__(
        self,
        obo_path: str,
        namespace_limits: Optional[Dict[str, int]] = None,
        semantic_clusters: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.hierarchy = GOHierarchy(obo_path)
        self.namespace_limits = dict(self.DEFAULT_LIMITS)
        if namespace_limits:
            self.namespace_limits.update(namespace_limits)
        self.semantic_clusters = semantic_clusters or self.DEFAULT_CLUSTERS
        self._depth_cache = {nid: node.depth for nid, node in self.hierarchy._nodes.items()}

    # ----------------------------
    # 基础判断
    # ----------------------------
    def is_ancestor(self, ancestor_go_id: str, child_go_id: str) -> bool:
        if ancestor_go_id == child_go_id:
            return False
        node = self.hierarchy._nodes.get(ancestor_go_id)
        if not node:
            return False
        return child_go_id in node.descendants

    def is_descendant(self, child_go_id: str, ancestor_go_id: str) -> bool:
        return self.is_ancestor(ancestor_go_id, child_go_id)

    def same_namespace(self, a: MergeCandidate, b: MergeCandidate) -> bool:
        return a.namespace == b.namespace

    def cluster_id(self, candidate: MergeCandidate) -> Optional[str]:
        name = candidate.go_name.lower().strip()
        for cid, members in self.semantic_clusters.items():
            if name in members:
                return cid
        return None

    # ----------------------------
    # 打分
    # ----------------------------
    def text_match_score(self, text: str, candidate: MergeCandidate) -> float:
        text_l = text.lower()
        name_l = candidate.go_name.lower().strip()
        span_l = candidate.span.lower().strip() if candidate.span else ""

        if name_l and name_l in text_l:
            return 1.0
        if span_l and span_l == name_l:
            return 0.95
        if span_l and span_l in name_l:
            return 0.75

        # 简单词重叠
        name_words = set(name_l.split())
        text_words = set(text_l.replace("/", " ").replace("-", " ").split())
        if name_words:
            overlap = len(name_words & text_words) / len(name_words)
            return min(0.8, overlap)
        return 0.0

    def specificity_score(self, candidate: MergeCandidate) -> float:
        depth = self._depth_cache.get(candidate.go_id, 0)
        # 深度越深，越具体；做一个轻量归一化
        return min(1.0, depth / 12.0)

    def main_concept_bonus(self, text: str, candidate: MergeCandidate) -> float:
        text_l = text.lower()
        name_l = candidate.go_name.lower()

        bonus = 0.0
        # 若术语名直接出现，偏向主概念
        if name_l in text_l:
            bonus += 0.15
        # 若 span 与标准名非常接近，再加一点
        if candidate.span and candidate.span.lower().strip() == name_l:
            bonus += 0.10
        return min(0.25, bonus)

    def final_score(self, text: str, candidate: MergeCandidate) -> float:
        text_score = self.text_match_score(text, candidate)
        spec_score = self.specificity_score(candidate)
        model_score = max(0.0, min(1.0, candidate.score))
        bonus = self.main_concept_bonus(text, candidate)

        score = (
            0.45 * text_score +
            0.25 * model_score +
            0.20 * spec_score +
            0.10 * bonus
        )
        return round(min(1.0, score), 4)

    # ----------------------------
    # 归并规则
    # ----------------------------
    def _dedupe_same_go(self, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        best: Dict[str, MergeCandidate] = {}
        for c in candidates:
            if c.go_id not in best or c.score > best[c.go_id].score:
                best[c.go_id] = c
        return list(best.values())

    def _resolve_hierarchy_conflicts(self, text: str, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        kept: List[MergeCandidate] = []
        for cand in sorted(candidates, key=lambda x: self.final_score(text, x), reverse=True):
            conflict = False
            for kept_c in kept:
                if not self.same_namespace(cand, kept_c):
                    continue
                if self.is_ancestor(cand.go_id, kept_c.go_id) or self.is_descendant(cand.go_id, kept_c.go_id):
                    conflict = True
                    # 若当前比已保留的更优，则替换
                    if self.final_score(text, cand) > self.final_score(text, kept_c):
                        kept.remove(kept_c)
                        kept.append(cand)
                    break
            if not conflict:
                kept.append(cand)
        return kept

    def _resolve_cluster_conflicts(self, text: str, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        groups: Dict[Tuple[str, str], List[MergeCandidate]] = {}
        no_cluster: List[MergeCandidate] = []

        for c in candidates:
            cid = self.cluster_id(c)
            if cid is None:
                no_cluster.append(c)
            else:
                groups.setdefault((c.namespace, cid), []).append(c)

        merged = list(no_cluster)
        for _, group in groups.items():
            best = max(group, key=lambda x: self.final_score(text, x))
            merged.append(best)
        return merged

    def _apply_namespace_limits(self, text: str, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        grouped: Dict[str, List[MergeCandidate]] = {}
        for c in candidates:
            grouped.setdefault(c.namespace, []).append(c)

        final: List[MergeCandidate] = []
        for namespace, items in grouped.items():
            limit = self.namespace_limits.get(namespace, len(items))
            items_sorted = sorted(items, key=lambda x: self.final_score(text, x), reverse=True)
            final.extend(items_sorted[:limit])
        return final

    # ----------------------------
    # 外部接口
    # ----------------------------
    def merge(self, text: str, candidates: Iterable[MergeCandidate]) -> List[MergedGO]:
        cand_list = self._dedupe_same_go(list(candidates))
        if not cand_list:
            return []

        stage1 = self._resolve_hierarchy_conflicts(text, cand_list)
        stage2 = self._resolve_cluster_conflicts(text, stage1)
        stage3 = self._apply_namespace_limits(text, stage2)

        kept_ids = {c.go_id for c in stage3}
        out: List[MergedGO] = []
        for c in sorted(stage3, key=lambda x: (x.namespace, -self.final_score(text, x), x.go_id)):
            dropped = [x.go_id for x in cand_list if x.go_id != c.go_id and x.go_id not in kept_ids]
            out.append(MergedGO(
                go_id=c.go_id,
                go_name=c.go_name,
                namespace=c.namespace,
                score=self.final_score(text, c),
                kept_from=[c],
                dropped_go_ids=dropped,
                reason="hierarchy+cluster+namespace_limit",
            ))
        return out

    def merge_go_ids(
        self,
        text: str,
        go_ids: Iterable[str],
        score: float = 1.0,
        source: str = "unknown",
    ) -> List[MergedGO]:
        candidates: List[MergeCandidate] = []
        for gid in go_ids:
            node = self.hierarchy._nodes.get(gid)
            if not node:
                continue
            candidates.append(MergeCandidate(
                go_id=gid,
                go_name=node.name,
                namespace=node.namespace,
                score=score,
                source=source,
                span=node.name,
                match_type="external",
            ))
        return self.merge(text, candidates)
