from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .config import settings
from .embedding import encode_texts


class VectorStore:
    def __init__(
        self,
        index: faiss.Index,
        metadata: Dict[int, Dict[str, str]],
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
    ):
        self.index = index
        self.metadata = metadata
        self.embedding_backend = embedding_backend
        self.embedding_model_name = embedding_model_name

    @classmethod
    def load(
        cls,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
    ) -> "VectorStore":
        resolved_index_path = Path(index_path or settings.faiss_index_path)
        resolved_metadata_path = Path(metadata_path or settings.faiss_metadata_path)

        if not resolved_index_path.exists() or not resolved_metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS 索引或元数据不存在，请先运行 `python -m app.index_builder` 构建索引。\n"
                f"期待索引: {resolved_index_path}\n期待元数据: {resolved_metadata_path}"
            )

        index = faiss.read_index(str(resolved_index_path))

        with resolved_metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dim = data.get("dim")
        if dim and index.d != dim:
            raise ValueError(
                f"索引维度 {index.d} 与元数据记录的维度 {dim} 不一致，请重新构建索引。"
            )

        items: Dict[int, Dict[str, str]] = {
            int(k): v for k, v in data.get("items", {}).items()
        }

        return cls(
            index=index,
            metadata=items,
            embedding_backend=embedding_backend,
            embedding_model_name=embedding_model_name,
        )

    @classmethod
    def load_ic(cls) -> "VectorStore":
        """加载 IC 压缩版 FAISS 索引（由 build_ic_index.py 预构建）。"""
        index_path    = Path(settings.ic_faiss_index_path)
        metadata_path = Path(settings.ic_faiss_metadata_path)

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"IC 压缩索引不存在，请先运行 `python build_ic_index.py` 构建。\n"
                f"期待索引: {index_path}\n期待元数据: {metadata_path}"
            )

        index = faiss.read_index(str(index_path))

        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        items: Dict[int, Dict[str, str]] = {
            int(k): v for k, v in data.get("items", {}).items()
        }

        print(f"[VectorStore] 已加载 IC 压缩索引：{len(items)} 个代表术语 <- {index_path}")
        return cls(index=index, metadata=items)

    def search(self, query: str, top_k: int) -> List[Tuple[float, Dict[str, str]]]:
        """对单个自然语言问题做语义检索，返回 (score, metadata) 列表。"""
        query_embedding = encode_texts(
            [query],
            backend=self.embedding_backend,
            model_name=self.embedding_model_name,
        )
        query_vec = query_embedding.astype("float32")

        distances, indices = self.index.search(query_vec, top_k)
        distances = distances[0]
        indices = indices[0]

        results: List[Tuple[float, Dict[str, str]]] = []
        for dist, idx in zip(distances, indices):
            if int(idx) not in self.metadata:
                continue
            meta = self.metadata[int(idx)]
            score = float(1.0 / (1.0 + dist))
            results.append((score, meta))

        return results


_vector_store: VectorStore | None = None


class TwoStageVectorStore:
    """
    两阶段检索：
      Stage 1 - 在 IC 压缩索引（~800 代表术语）中粗筛 Top-coarse_k
      Stage 2 - 收集这些代表节点的所有子孙术语，在全量索引中精细检索 Top-k

    优点：速度接近 IC 压缩检索，准确率接近全量检索。
    """

    def __init__(
        self,
        full_store: VectorStore,
        ic_store: VectorStore,
        descendants: Dict[str, List[str]],  # {代表节点 go_id: [子孙 go_id, ...]}
        coarse_k: int = 10,
    ):
        self.full_store  = full_store
        self.ic_store    = ic_store
        self.descendants = descendants
        self.coarse_k    = coarse_k

        # 构建全量索引的 go_id -> idx 反查表，加速 Stage 2 过滤
        self._go_to_idx: Dict[str, int] = {}
        for idx, meta in full_store.metadata.items():
            gid = meta.get("go_id", "")
            if gid:
                self._go_to_idx[gid] = idx

    def search(self, query: str, top_k: int) -> List[Tuple[float, Dict[str, str]]]:
        # ── Stage 1：IC 压缩粗筛 ──────────────────────────────────────────
        coarse_hits = self.ic_store.search(query, top_k=self.coarse_k)
        if not coarse_hits:
            # 降级到全量检索
            return self.full_store.search(query, top_k=top_k)

        # 收集候选子孙 go_id 集合（代表节点本身 + 其所有子孙）
        candidate_go_ids: set[str] = set()
        for _, meta in coarse_hits:
            rep_id = meta.get("go_id", "")
            if rep_id:
                candidate_go_ids.add(rep_id)
                candidate_go_ids.update(self.descendants.get(rep_id, []))

        # ── Stage 2：在全量索引中只对候选集做精细检索 ─────────────────────
        # 把候选 go_id 转成全量索引的行号集合
        candidate_idxs: set[int] = {
            self._go_to_idx[gid]
            for gid in candidate_go_ids
            if gid in self._go_to_idx
        }

        if not candidate_idxs:
            # 候选集为空（代表节点没有子孙在全量索引中）→ 直接返回粗筛结果
            return coarse_hits[:top_k]

        # 向量化查询
        query_vec = encode_texts([query]).astype("float32")

        # 构建临时子索引（只含候选行，使用 L2 与全量索引一致）
        candidate_list = sorted(candidate_idxs)
        dim = self.full_store.index.d
        sub_index = faiss.IndexFlatL2(dim)

        # 从全量索引中取出候选向量
        all_vecs = np.zeros((len(candidate_list), dim), dtype="float32")
        for i, idx in enumerate(candidate_list):
            self.full_store.index.reconstruct(idx, all_vecs[i])
        sub_index.add(all_vecs)

        k2 = min(top_k, len(candidate_list))
        dists, sub_idxs = sub_index.search(query_vec, k2)

        results: List[Tuple[float, Dict[str, str]]] = []
        for dist, sub_idx in zip(dists[0], sub_idxs[0]):
            if sub_idx < 0 or sub_idx >= len(candidate_list):
                continue
            real_idx = candidate_list[sub_idx]
            meta = self.full_store.metadata.get(real_idx)
            if meta is None:
                continue
            # 与全量检索保持一致：1/(1+L2距离) 映射到 0~1
            score = float(1.0 / (1.0 + dist))
            results.append((score, meta))

        return results


_vector_store: VectorStore | TwoStageVectorStore | None = None


def _load_descendants() -> Dict[str, List[str]]:
    desc_path = Path(settings.ic_faiss_index_path).parent / "go_ic_descendants.json"
    if not desc_path.exists():
        print(f"[VectorStore] 未找到子孙映射文件 {desc_path}，两阶段检索将退化为粗筛")
        return {}
    with desc_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_vector_store(profile: str | None = None) -> VectorStore | TwoStageVectorStore:
    global _vector_store
    selected_profile = settings.get_profile(profile, purpose="rag")
    cache_key = (
        selected_profile["faiss_index_path"],
        selected_profile["faiss_metadata_path"],
        selected_profile["embedding_backend"],
        selected_profile["embedding_model_name"],
        settings.use_ic_index,
    )
    current_key = getattr(_vector_store, "_cache_key", None) if _vector_store is not None else None

    if _vector_store is None or current_key != cache_key:
        if settings.use_ic_index:
            ic_index_path = Path(settings.ic_faiss_index_path)
            full_index_path = Path(selected_profile["faiss_index_path"])
            if ic_index_path.exists() and full_index_path.exists():
                print("[VectorStore] 构建两阶段检索器（IC粗筛 + 全量精筛）...")
                ic_store = VectorStore.load_ic()
                full_store = VectorStore.load(
                    index_path=selected_profile["faiss_index_path"],
                    metadata_path=selected_profile["faiss_metadata_path"],
                    embedding_backend=selected_profile["embedding_backend"],
                    embedding_model_name=selected_profile["embedding_model_name"],
                )
                descendants = _load_descendants()
                _vector_store = TwoStageVectorStore(
                    full_store=full_store,
                    ic_store=ic_store,
                    descendants=descendants,
                    coarse_k=max(10, settings.top_k * 3),
                )
                print(f"[VectorStore] 两阶段检索就绪：IC {len(ic_store.metadata)} 代表术语，"
                      f"全量 {len(full_store.metadata)} 术语")
            else:
                print("[VectorStore] 全量索引不存在，降级到纯 IC 压缩检索")
                _vector_store = VectorStore.load_ic()
        else:
            _vector_store = VectorStore.load(
                index_path=selected_profile["faiss_index_path"],
                metadata_path=selected_profile["faiss_metadata_path"],
                embedding_backend=selected_profile["embedding_backend"],
                embedding_model_name=selected_profile["embedding_model_name"],
            )
        setattr(_vector_store, "_cache_key", cache_key)
    return _vector_store


def reset_vector_store() -> None:
    """清空缓存，下次调用 get_vector_store() 时重新加载（切换索引后调用）。"""
    global _vector_store
    _vector_store = None

