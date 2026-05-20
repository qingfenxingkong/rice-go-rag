from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import settings
from . import embedding_offline, embedding_sbert


def encode_texts(
    texts: Iterable[str],
    batch_size: int = 32,
    backend: str | None = None,
    model_name: str | None = None,
) -> np.ndarray:
    """根据配置选择 embedding 版本。

    - EMBEDDING_BACKEND=offline（默认）：使用无需外网的 hash 向量。
    - EMBEDDING_BACKEND=sbert：使用 SentenceTransformer（需要能访问 huggingface 或本地已有缓存）。
    - 可通过 backend / model_name 显式覆盖默认配置，用于 NER 与 RAG 分开切换 profile。
    """
    selected_backend = (backend or settings.embedding_backend or "offline").lower()
    if selected_backend == "sbert":
        return embedding_sbert.encode_texts(texts, batch_size=batch_size, model_name=model_name)
    # 默认走离线版本，保证在无网络环境下也能运行
    return embedding_offline.encode_texts(texts, batch_size=batch_size)

