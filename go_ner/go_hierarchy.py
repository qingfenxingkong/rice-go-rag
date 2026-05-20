"""
go_ner/go_hierarchy.py
GO 层级压缩算法

核心思路：
  1. 从 OBO 解析 is_a 关系，构建 DAG
  2. 计算每个节点的子孙覆盖度
  3. 贪心算法选出最优代表性父节点集合
  4. 压缩后的术语集合供 DictNER/VectorNER/LLMNER 使用
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .obo_parser import GOEntry, parse_obo


@dataclass
class GONode:
    entry: GOEntry
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    depth: int = 0
    descendants: Set[str] = field(default_factory=set)

    @property
    def go_id(self) -> str: return self.entry.go_id
    @property
    def name(self) -> str: return self.entry.name
    @property
    def namespace(self) -> str: return self.entry.namespace
    @property
    def descendant_count(self) -> int: return len(self.descendants)


class GOHierarchy:
    """
    GO DAG 及层级压缩算法。

    用法：
        hier = GOHierarchy("go.obo")
        nodes = hier.select_representative_nodes(max_nodes=300)
        compressed = hier.get_compressed_dict(max_nodes=300)
    """

    ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}

    def __init__(self, obo_path: str) -> None:
        self._obo_path = str(obo_path)
        print("[GOHierarchy] 解析 OBO 文件...")
        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)
        print("[GOHierarchy] 构建 DAG...")
        self._nodes: Dict[str, GONode] = {}
        self._build_dag()
        print("[GOHierarchy] 计算节点深度...")
        self._compute_depths()
        print("[GOHierarchy] 计算子孙集合（耗时较长）...")
        self._compute_descendants()
        print(f"[GOHierarchy] 完成，共 {len(self._nodes)} 个节点")

    def _build_dag(self) -> None:
        for go_id, entry in self._entries.items():
            if go_id == entry.go_id:
                self._nodes[go_id] = GONode(entry=entry)
        # 解析 is_a 关系
        current_id = None
        with open(self._obo_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip()
                if line == "[Term]":
                    current_id = None
                elif line.startswith("id: "):
                    current_id = line[4:].strip()
                elif line.startswith("is_a: ") and current_id:
                    parent_id = line[6:].split("!")[0].strip()
                    if current_id in self._nodes and parent_id in self._nodes:
                        self._nodes[current_id].parents.add(parent_id)
                        self._nodes[parent_id].children.add(current_id)

    def _compute_depths(self) -> None:
        queue = deque()
        visited = set()
        for root_id in self.ROOTS:
            if root_id in self._nodes:
                self._nodes[root_id].depth = 0
                queue.append(root_id)
                visited.add(root_id)
        while queue:
            cur = queue.popleft()
            for child_id in self._nodes[cur].children:
                if child_id not in visited:
                    visited.add(child_id)
                    self._nodes[child_id].depth = self._nodes[cur].depth + 1
                    queue.append(child_id)

    def _compute_descendants(self) -> None:
        """从叶节点向上传播，计算每个节点的所有子孙集合。"""
        # 计算每个节点还有多少子节点未处理
        remaining = {nid: len(n.children) for nid, n in self._nodes.items()}
        queue = deque([nid for nid, r in remaining.items() if r == 0])
        processed: Set[str] = set()

        while queue:
            nid = queue.popleft()
            if nid in processed:
                continue
            processed.add(nid)
            node = self._nodes[nid]
            for child_id in node.children:
                node.descendants.add(child_id)
                node.descendants.update(self._nodes[child_id].descendants)
            for parent_id in node.parents:
                if parent_id in self._nodes:
                    remaining[parent_id] -= 1
                    if remaining[parent_id] == 0:
                        queue.append(parent_id)

    # ── 信息增益（IC）计算 ────────────────────────────────────────────

    def compute_ic_structural(self) -> Dict[str, float]:
        """
        方案 A：结构性信息增益（无需基因注释，使用 Laplace 平滑）

        IC_struct(t) = -log2( (|descendants(t)| + 1) / (|total_terms| + |total_terms|) )

        叶节点（子孙数=0）IC 最高（最具体）
        根节点（子孙数最大）IC 最低（最宽泛）
        """
        import math
        total = len(self._nodes)
        smooth_denom = total + total  # Laplace 平滑分母
        ic: Dict[str, float] = {}
        for nid, node in self._nodes.items():
            p = (node.descendant_count + 1) / smooth_denom
            ic[nid] = -math.log2(p)
        print(f"[GOHierarchy] 结构性 IC 计算完成（Laplace 平滑），"
              f"范围: {min(ic.values()):.2f} ~ {max(ic.values()):.2f}")
        return ic

    def compute_ic_from_neo4j(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str,
    ) -> tuple:
        """
        方案 B：基于 Neo4j Gene->GO_Term 注释的真实信息增益

        Returns:
            (ic_dict, annotation_counts)
            ic_dict:           {go_id: IC值}
            annotation_counts: {go_id: 传播后注释基因数}
        """
        import math
        from neo4j import GraphDatabase

        print("[GOHierarchy] 从 Neo4j 获取基因注释数据...")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        direct_annotations: Dict[str, Set[str]] = defaultdict(set)  # go_id -> {gene_names}

        try:
            with driver.session(database=neo4j_database) as session:
                result = session.run("""
                    MATCH (g:Gene)-[:GO_Mention]->(go:GO_Term)
                    RETURN go.GO_Term AS go_id, g.name AS gene_name
                """)
                for rec in result:
                    go_id = rec["go_id"]
                    gene = rec["gene_name"]
                    if go_id and gene:
                        direct_annotations[go_id].add(gene)
        finally:
            driver.close()

        total_annotated = sum(len(v) for v in direct_annotations.values())
        print(f"[GOHierarchy] 获取到 {len(direct_annotations)} 个 GO_Term 的注释，"
              f"共 {total_annotated} 条 Gene->GO 关系")

        if total_annotated == 0:
            print("[GOHierarchy] 警告：Neo4j 中无注释数据，降级到结构性 IC")
            ic = self.compute_ic_structural()
            return ic, {}

        # 向上传播：每个节点的有效基因集 = 直接注释 + 所有子孙的注释（去重）
        print("[GOHierarchy] 向上传播注释基因集...")
        propagated: Dict[str, Set[str]] = {nid: set(direct_annotations.get(nid, set()))
                                            for nid in self._nodes}
        remaining = {nid: len(n.children) for nid, n in self._nodes.items()}
        queue: deque = deque([nid for nid, r in remaining.items() if r == 0])
        processed: Set[str] = set()

        while queue:
            nid = queue.popleft()
            if nid in processed:
                continue
            processed.add(nid)
            for parent_id in self._nodes[nid].parents:
                if parent_id in self._nodes:
                    propagated[parent_id].update(propagated[nid])
                    remaining[parent_id] -= 1
                    if remaining[parent_id] == 0:
                        queue.append(parent_id)

        # Laplace 平滑计算 IC，所有节点都有连续 IC 值
        # IC(t) = -log2( (genes(t)+1) / (total_genes + total_terms) )
        total_genes = max(len(set().union(*propagated.values())), 1)
        total_terms = len(self._nodes)
        smooth_denom = total_genes + total_terms
        print(f"[GOHierarchy] 总唯一基因数: {total_genes}, Laplace 分母: {smooth_denom}")

        ic: Dict[str, float] = {}
        for nid, genes in propagated.items():
            p = (len(genes) + 1) / smooth_denom
            ic[nid] = -math.log2(p)

        # 注释基因数字典
        annotation_counts: Dict[str, int] = {nid: len(genes) for nid, genes in propagated.items()}

        annotated_nodes = sum(1 for g in propagated.values() if g)
        print(f"[GOHierarchy] IC 计算完成（Laplace 平滑），有注释节点: {annotated_nodes}，"
              f"IC 范围: {min(ic.values()):.2f} ~ {max(ic.values()):.2f}")
        return ic, annotation_counts

    def select_representative_nodes_by_ic(
        self,
        ic: Dict[str, float],
        namespace: Optional[str] = None,
        min_depth: int = 2,
        max_depth: int = 6,
        min_descendants: int = 5,
        max_nodes: int = 500,
        coverage_threshold: float = 0.95,
        ic_min: float = 1.0,
        ic_max: float = 8.0,
        min_annotations: int = 0,
        annotation_counts: Optional[Dict[str, int]] = None,
    ) -> List[GONode]:
        """
        基于信息增益的动态贪心代表节点选择。

        打分公式：
          score(t) = IC(t) * log2(新增覆盖叶节点数 + 1)

        每轮动态重新计算，保证全局最优。

        Args:
            ic:                   {go_id: IC值} 字典
            namespace:            限定命名空间
            min_depth/max_depth:  深度范围
            min_descendants:      最少子孙数
            max_nodes:            最多选出多少节点
            coverage_threshold:   叶节点覆盖率阈值
            ic_min/ic_max:        IC 值范围过滤
            min_annotations:      最少注释基因数（过滤注释太稀疏的节点，需配合 annotation_counts）
            annotation_counts:    {go_id: 注释基因数}，用于 min_annotations 过滤
        """
        import math

        # 候选节点
        candidates = []
        for node in self._nodes.values():
            if namespace is not None and node.namespace != namespace:
                continue
            if not (min_depth <= node.depth <= max_depth):
                continue
            if node.descendant_count < min_descendants:
                continue
            node_ic = ic.get(node.go_id, 0)
            if not (ic_min <= node_ic <= ic_max):
                continue
            if min_annotations > 0 and annotation_counts is not None:
                ann = annotation_counts.get(node.go_id, 0)
                if ann < min_annotations:
                    continue
            candidates.append(node)

        all_leaves: Set[str] = set(
            nid for nid, node in self._nodes.items()
            if (namespace is None or node.namespace == namespace)
            and len(node.children) == 0
        )
        total_leaves = max(len(all_leaves), 1)
        print(f"[GOHierarchy][IC] 候选: {len(candidates)}, 叶节点: {total_leaves}")

        selected: List[GONode] = []
        covered: Set[str] = set()
        cand_set = list(candidates)

        # 动态贪心：每轮重新计算新增覆盖
        for _ in range(max_nodes):
            if len(covered) / total_leaves >= coverage_threshold:
                break
            if not cand_set:
                break

            best_score = -1.0
            best_node = None
            best_new_leaves = set()

            for node in cand_set:
                new_leaves = (node.descendants & all_leaves) - covered
                node_ic = ic.get(node.go_id, 0.0)
                # 打分：IC * log2(新增覆盖+1)
                s = node_ic * math.log2(len(new_leaves) + 1)
                if s > best_score:
                    best_score = s
                    best_node = node
                    best_new_leaves = new_leaves

            if best_node is None or best_score <= 0:
                break

            selected.append(best_node)
            covered.update(best_new_leaves)
            if best_node.go_id in all_leaves:
                covered.add(best_node.go_id)
            cand_set.remove(best_node)

        final_cov = len(covered) / total_leaves
        print(
            f"[GOHierarchy][IC] 选出 {len(selected)} 个代表节点，"
            f"覆盖 {len(covered)}/{total_leaves} ({final_cov*100:.1f}%) 叶节点"
        )
        return selected

    def get_compressed_dict_by_ic(
        self,
        use_neo4j: bool = True,
        neo4j_uri: str = "",
        neo4j_user: str = "",
        neo4j_password: str = "",
        neo4j_database: str = "",
        namespace: Optional[str] = None,
        min_depth: int = 2,
        max_depth: int = 6,
        min_descendants: int = 5,
        max_nodes: int = 500,
        coverage_threshold: float = 0.95,
        ic_min: float = 1.0,
        ic_max: float = 8.0,
        min_annotations: int = 3,
    ) -> Dict[str, GOEntry]:
        """
        一键获取基于 IC 的压缩字典。
        优先用 Neo4j 真实注释，不可用时降级到结构性 IC。
        """
        annotation_counts: Dict[str, int] = {}
        if use_neo4j and neo4j_uri:
            try:
                ic, annotation_counts = self.compute_ic_from_neo4j(
                    neo4j_uri, neo4j_user, neo4j_password, neo4j_database
                )
            except Exception as e:
                print(f"[GOHierarchy] Neo4j IC 计算失败({e})，降级到结构性 IC")
                ic = self.compute_ic_structural()
        else:
            ic = self.compute_ic_structural()

        nodes = self.select_representative_nodes_by_ic(
            ic=ic, namespace=namespace,
            min_depth=min_depth, max_depth=max_depth,
            min_descendants=min_descendants, max_nodes=max_nodes,
            coverage_threshold=coverage_threshold,
            ic_min=ic_min, ic_max=ic_max,
            min_annotations=min_annotations,
            annotation_counts=annotation_counts if annotation_counts else None,
        )
        return {n.go_id: n.entry for n in nodes}

    def select_representative_nodes(
        self,
        namespace: Optional[str] = None,
        min_depth: int = 2,
        max_depth: int = 6,
        min_descendants: int = 10,
        max_nodes: int = 500,
        coverage_threshold: float = 0.95,
    ) -> List[GONode]:
        """
        贪心算法选出最优代表性父节点集合。

        策略：
          1. 筛选满足深度/子孙数约束的候选节点
          2. 按「覆盖叶节点数 / 深度惩罚」打分排序
          3. 贪心选择：每次选覆盖最多未覆盖叶节点的节点
          4. 达到 coverage_threshold 或 max_nodes 时停止

        Args:
            namespace:           限定命名空间，None 表示全部
            min_depth:           节点最小深度（过滤过于宽泛的根节点）
            max_depth:           节点最大深度（过滤过于具体的叶节点）
            min_descendants:     最少子孙数
            max_nodes:           最多选出多少个代表节点
            coverage_threshold:  叶节点覆盖率阈值

        Returns:
            代表性 GONode 列表（按覆盖度降序）
        """
        candidates = [
            node for node in self._nodes.values()
            if (namespace is None or node.namespace == namespace)
            and min_depth <= node.depth <= max_depth
            and node.descendant_count >= min_descendants
        ]

        if not candidates:
            print("[GOHierarchy] 无满足条件的候选节点，返回全部")
            candidates = list(self._nodes.values())

        # 所有叶节点
        all_leaves: Set[str] = set(
            nid for nid, node in self._nodes.items()
            if (namespace is None or node.namespace == namespace)
            and len(node.children) == 0
        )
        total_leaves = max(len(all_leaves), 1)
        print(f"[GOHierarchy] 候选: {len(candidates)}, 叶节点: {total_leaves}")

        # 打分：覆盖叶节点数，加适度深度偏好
        ideal_depth = (min_depth + max_depth) / 2
        def score(node: GONode) -> float:
            leaf_cov = len(node.descendants & all_leaves)
            depth_pen = abs(node.depth - ideal_depth) * 0.05
            return leaf_cov / (1 + depth_pen)

        candidates.sort(key=score, reverse=True)

        # 贪心覆盖
        selected: List[GONode] = []
        covered: Set[str] = set()

        for node in candidates:
            if len(selected) >= max_nodes:
                break
            if len(covered) / total_leaves >= coverage_threshold:
                break
            new = (node.descendants & all_leaves) - covered
            if not new and node.go_id not in all_leaves:
                continue
            selected.append(node)
            covered.update(new)
            if node.go_id in all_leaves:
                covered.add(node.go_id)

        print(
            f"[GOHierarchy] 选出 {len(selected)} 个代表节点，"
            f"覆盖 {len(covered)}/{total_leaves} ({len(covered)/total_leaves*100:.1f}%) 叶节点"
        )
        return selected

    def get_compressed_dict(
        self,
        namespace: Optional[str] = None,
        min_depth: int = 2,
        max_depth: int = 6,
        min_descendants: int = 10,
        max_nodes: int = 500,
        coverage_threshold: float = 0.95,
    ) -> Dict[str, GOEntry]:
        """返回压缩后的 {go_id: GOEntry} 字典，可直接传给 DictNER/VectorNER。"""
        nodes = self.select_representative_nodes(
            namespace=namespace, min_depth=min_depth, max_depth=max_depth,
            min_descendants=min_descendants, max_nodes=max_nodes,
            coverage_threshold=coverage_threshold,
        )
        return {n.go_id: n.entry for n in nodes}

    def find_best_ancestor(self, go_id: str, max_depth: int = 5) -> Optional[GONode]:
        """给定具体 GO ID，找到其在 max_depth 以内最佳概括性祖先。"""
        if go_id not in self._nodes:
            return None
        node = self._nodes[go_id]
        if node.depth <= max_depth:
            return node
        # BFS 向上找最近的满足深度的祖先
        queue: deque = deque(node.parents)
        visited: Set[str] = set(node.parents)
        while queue:
            pid = queue.popleft()
            p = self._nodes.get(pid)
            if p and p.depth <= max_depth:
                return p
            if p:
                for gp in p.parents:
                    if gp not in visited:
                        visited.add(gp)
                        queue.append(gp)
        return node

    def get_node(self, go_id: str) -> Optional[GONode]:
        return self._nodes.get(go_id)

    def get_ancestors(self, go_id: str) -> List[GONode]:
        """返回某节点的所有祖先节点，按深度升序。"""
        if go_id not in self._nodes:
            return []
        visited: Set[str] = set()
        queue: deque = deque(self._nodes[go_id].parents)
        ancestors = []
        while queue:
            pid = queue.popleft()
            if pid in visited:
                continue
            visited.add(pid)
            p = self._nodes.get(pid)
            if p:
                ancestors.append(p)
                queue.extend(p.parents)
        ancestors.sort(key=lambda n: n.depth)
        return ancestors

    def print_stats(self) -> None:
        """打印 DAG 统计信息。"""
        ns_counts: Dict[str, int] = defaultdict(int)
        depth_dist: Dict[int, int] = defaultdict(int)
        for node in self._nodes.values():
            ns_counts[node.namespace] += 1
            depth_dist[node.depth] += 1

        print("\n── GO DAG 统计 ──")
        for ns, cnt in sorted(ns_counts.items()):
            print(f"  {ns}: {cnt} 个节点")
        print(f"  总计: {len(self._nodes)} 个节点")
        print("\n  深度分布（前10层）:")
        for d in sorted(depth_dist)[:10]:
            bar = '█' * min(depth_dist[d] // 50, 40)
            print(f"  深度 {d:2d}: {depth_dist[d]:5d} {bar}")

    def show_compressed_examples(
        self,
        namespace: str = "biological_process",
        max_nodes: int = 20,
    ) -> None:
        """展示压缩结果示例。"""
        nodes = self.select_representative_nodes(
            namespace=namespace, max_nodes=max_nodes
        )
        print(f"\n── 压缩代表节点示例（{namespace}）──")
        for n in nodes[:max_nodes]:
            print(
                f"  [{n.go_id}] 深度={n.depth:2d} 子孙={n.descendant_count:5d}  {n.name}"
            )
