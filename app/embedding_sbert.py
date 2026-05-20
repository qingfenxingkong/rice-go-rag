from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=8)
def get_sentence_model(model_name: str | None = None) -> SentenceTransformer:
    """懒加载 Sentence Transformer 模型（需要外网或本地已缓存模型）。"""
    model_path = model_name or settings.embedding_model_name
    # 如果给的是本地路径，强制只从本地读取，避免任何联网请求
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path, local_files_only=True)
    else:
        model = SentenceTransformer(model_path)
    return model


def encode_texts(texts: Iterable[str], batch_size: int = 32, model_name: str | None = None) -> np.ndarray:
    """使用 SentenceTransformer 的高质量语义向量版本。"""
    model = get_sentence_model(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings

