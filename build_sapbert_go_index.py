from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from go_ner.obo_parser import GOEntry, parse_obo


def _collect_unique_entries(entries: Dict[str, GOEntry]) -> List[GOEntry]:
    seen: set[str] = set()
    unique_entries: List[GOEntry] = []
    for entry in entries.values():
        if entry.go_id in seen:
            continue
        seen.add(entry.go_id)
        unique_entries.append(entry)
    return unique_entries


def _build_text(entry: GOEntry, include_synonyms: bool) -> str:
    parts = [entry.name]
    if include_synonyms and entry.synonyms:
        parts.append("Synonyms: " + "; ".join(entry.synonyms))
    if entry.definition:
        parts.append(entry.definition)
    return ". ".join(part.strip() for part in parts if part and part.strip())


def build_sapbert_go_index(
    obo_path: str | Path,
    model_name: str,
    index_path: str | Path,
    metadata_path: str | Path,
    batch_size: int = 128,
    include_synonyms: bool = True,
) -> None:
    obo_path = Path(obo_path)
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    print(f"[SapBERT Index] 读取 OBO: {obo_path}")
    entries = parse_obo(obo_path)
    unique_entries = _collect_unique_entries(entries)
    print(f"[SapBERT Index] 唯一 GO 主术语数: {len(unique_entries)}")

    texts: List[str] = []
    metadata_items: Dict[str, Dict[str, str]] = {}

    for i, entry in enumerate(unique_entries):
        texts.append(_build_text(entry, include_synonyms=include_synonyms))
        metadata_items[str(i)] = {
            "go_id": entry.go_id,
            "name": entry.name,
            "namespace": entry.namespace,
            "description": entry.definition,
            "synonyms": "; ".join(entry.synonyms),
        }

    print(f"[SapBERT Index] 加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[SapBERT Index] 开始向量化 {len(texts)} 个 GO 术语...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings 必须是二维数组。")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SapBERT Index] 写入索引: {index_path}")
    faiss.write_index(index, str(index_path))

    metadata = {
        "dim": dim,
        "model_name": model_name,
        "index_type": "IndexFlatIP",
        "normalized": True,
        "source": str(obo_path),
        "include_synonyms": include_synonyms,
        "term_count": len(unique_entries),
        "items": metadata_items,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"[SapBERT Index] 构建完成：{len(unique_entries)} 个 GO 主术语，"
        f"dim={dim}，索引未覆盖旧文件。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 VectorNER 构建 SapBERT 专属 GO-only FAISS 索引，不覆盖旧索引。"
    )
    parser.add_argument("--obo", default="go.obo", help="go.obo 文件路径")
    parser.add_argument(
        "--model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="SentenceTransformer / HuggingFace 模型名或本地路径",
    )
    parser.add_argument(
        "--index-path",
        default="data/go_faiss_sapbert.index",
        help="输出 FAISS 索引路径",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/go_metadata_sapbert.json",
        help="输出元数据 JSON 路径",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="向量化 batch size")
    parser.add_argument(
        "--no-synonyms",
        action="store_true",
        help="构建文本时不拼接同义词",
    )
    args = parser.parse_args()

    if Path(args.index_path).exists() or Path(args.metadata_path).exists():
        print("[SapBERT Index] 提示：目标 SapBERT 索引文件已存在，将直接覆盖 SapBERT 新索引文件本身。")
        print("[SapBERT Index] 不会影响现有的 go_faiss.index / go_faiss_minilm.index 等旧索引。")

    build_sapbert_go_index(
        obo_path=args.obo,
        model_name=args.model,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        include_synonyms=not args.no_synonyms,
    )


if __name__ == "__main__":
    main()
