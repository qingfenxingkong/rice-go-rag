"""
go_ner/vector_ner.py
方案二：基于向量语义相似度的 GO 术语 NER + 概念标准化

策略：
  1. 用 SentenceTransformer 将文本分句/分段后编码
  2. 在预构建的 GO 术语向量库中检索 Top-K 最相似术语
  3. 适合处理非标准描述、同义表达、跨语言文本
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .obo_parser import GOEntry, parse_obo


@dataclass
class VectorNERResult:
    """向量检索结果。"""
    query: str          # 查询文本（句子或短语）
    go_id: str
    go_name: str
    namespace: str
    definition: str
    similarity: float   # 余弦相似度 (0~1)


class VectorNER:
    """
    基于语义向量的 GO 术语检索与标准化。

    Args:
        obo_path:       go.obo 文件路径
        model_path:     SentenceTransformer 模型路径（本地）
        index_path:     预构建的 FAISS 索引路径（可选，若无则自动构建）
        metadata_path:  索引对应的元数据 JSON 路径
        top_k:          每次检索返回的候选数量
    """

    def __init__(
        self,
        obo_path: str,
        model_path: str,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k

        print("[VectorNER] 加载 OBO 文件...")
        self._entries: Dict[str, GOEntry] = parse_obo(obo_path)

        print("[VectorNER] 加载向量模型...")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_path)

        # 尝试加载已有 FAISS 索引
        if index_path and metadata_path and Path(index_path).exists() and Path(metadata_path).exists():
            print(f"[VectorNER] 加载已有索引: {index_path}")
            import faiss
            self._index = faiss.read_index(index_path)
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._meta_items = meta["items"]  # {str(i): {...}}
            print(f"[VectorNER] 索引加载完成，共 {self._index.ntotal} 条")
        else:
            print("[VectorNER] 未找到预构建索引，从 OBO 构建临时索引（仅含名称+定义）...")
            self._index, self._meta_items = self._build_temp_index()

    def _build_temp_index(self):
        """从 OBO 数据临时构建向量索引（子集，不写磁盘）。"""
        import faiss

        texts = []
        meta_items = {}
        valid_entries = [e for e in self._entries.values() if e.go_id == e.go_id]
        # 去重（只处理主 ID）
        seen = set()
        unique_entries = []
        for e in self._entries.values():
            if e.go_id not in seen:
                seen.add(e.go_id)
                unique_entries.append(e)

        for i, entry in enumerate(unique_entries):
            text = f"{entry.name}. {entry.definition}"
            texts.append(text)
            meta_items[str(i)] = {
                "go_id": entry.go_id,
                "name": entry.name,
                "namespace": entry.namespace,
                "description": entry.definition,
            }

        print(f"[VectorNER] 向量化 {len(texts)} 个术语（首次需要较长时间）...")
        embeddings = self._model.encode(texts, batch_size=256, show_progress_bar=True)
        embeddings = embeddings.astype("float32")

        # 归一化用于余弦相似度
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalize
        index.add(embeddings)

        return index, meta_items

    def _encode_query(self, text: str) -> np.ndarray:
        import faiss
        vec = self._model.encode([text], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: Optional[int] = None) -> List[VectorNERResult]:
        """
        在 GO 术语库中语义检索与 query 最相关的术语。

        Args:
            query: 查询文本（一句话或一个描述片段）
            top_k: 返回数量，默认使用初始化时的 top_k
        """
        k = top_k or self.top_k
        vec = self._encode_query(query)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta_items.get(str(idx), {})
            go_id = meta.get("go_id", "")
            entry = self._entries.get(go_id)
            results.append(VectorNERResult(
                query=query,
                go_id=go_id,
                go_name=meta.get("name", ""),
                namespace=meta.get("namespace", ""),
                definition=meta.get("description", ""),
                similarity=round(float(score), 4),
            ))
        return results

    def recognize_sentences(
        self,
        text: str,
        sent_sep: str = r'[.;\n]',
        top_k: int = 3,
    ) -> Dict[str, List[VectorNERResult]]:
        """
        将文本按句子分割，对每个句子做语义检索。

        Returns:
            {句子: [VectorNERResult, ...]}
        """
        import re
        sentences = [s.strip() for s in re.split(sent_sep, text) if len(s.strip()) > 5]
        results = {}
        for sent in sentences:
            results[sent] = self.search(sent, top_k=top_k)
        return results

    def normalize(self, term: str) -> Optional[VectorNERResult]:
        """将一个术语/描述标准化为最匹配的 GO 术语。"""
        results = self.search(term, top_k=1)
        return results[0] if results else None

    def format_results(
        self,
        results: Dict[str, List[VectorNERResult]],
        threshold: float = 0.5,
    ) -> str:
        """格式化 recognize_sentences 的结果。"""
        lines = []
        for sent, hits in results.items():
            lines.append(f"句子：{sent}")
            filtered = [h for h in hits if h.similarity >= threshold]
            if not filtered:
                lines.append("  (未找到相似度达标的 GO 术语)")
            for h in filtered:
                lines.append(
                    f"  [{h.similarity:.3f}] {h.go_id} | {h.go_name}\n"
                    f"           命名空间: {h.namespace}\n"
                    f"           定义: {h.definition[:100]}..."
                )
            lines.append("")
        return "\n".join(lines)
