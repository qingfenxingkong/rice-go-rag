from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


EMBED_DIM = 256


def _embed_single(text: str) -> np.ndarray:
    """使用简单的 hashing trick 将文本编码为固定维度向量。

    完全离线，不依赖 huggingface 或任何预训练模型。
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    tokens = [t for t in text.lower().replace("\n", " ").split(" ") if t]
    if not tokens:
        return vec

    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16)
        idx = bucket % EMBED_DIM
        sign_bit = (bucket >> 1) & 1
        sign = 1.0 if sign_bit == 0 else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:  # noqa: ARG002
    """简易离线 embedding 版本。"""
    vecs = [_embed_single(t or "") for t in texts]
    return np.stack(vecs, axis=0)

