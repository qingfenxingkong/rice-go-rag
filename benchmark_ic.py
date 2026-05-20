from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Set

from app.config import settings
from app.vector_store import get_vector_store, reset_vector_store


def load_cases(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", []) if isinstance(data, dict) else []
    norm = []
    for c in cases:
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        gold_raw = c.get("gold", [])
        gold: Set[str] = set()
        if isinstance(gold_raw, list):
            for g in gold_raw:
                gs = str(g).strip().upper()
                if gs.startswith("GO:"):
                    gold.add(gs)
        norm.append({
            "id": c.get("id"),
            "text": text,
            "gold": gold,
            "lang": c.get("lang", "unknown"),
            "difficulty": c.get("difficulty", "unknown"),
        })
    return norm


def dcg_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    dcg = 0.0
    for i, gid in enumerate(pred[:k], start=1):
        rel = 1.0 if gid in gold else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    ideal_len = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(pred, gold, k) / idcg


def evaluate_mode(cases: List[dict], top_k: int, use_ic: bool, repeats: int) -> Dict[str, float]:
    settings.use_ic_index = use_ic
    # TwoStageVectorStore 的 coarse_k 由 settings.top_k 派生，
    # 为保证消融测试公平，这里同步成当前评测 top_k。
    settings.top_k = top_k

    hit_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    ndcg_scores: List[float] = []
    p_at_k_scores: List[float] = []
    latencies_ms: List[float] = []

    pos_cases = [c for c in cases if c["gold"]]
    neg_cases = [c for c in cases if not c["gold"]]

    neg_non_go_rate_runs: List[float] = []

    for _ in range(repeats):
        reset_vector_store()
        store = get_vector_store()

        neg_non_go = 0
        for case in pos_cases + neg_cases:
            q = case["text"]
            gold = case["gold"]

            t0 = time.perf_counter()
            hits = store.search(q, top_k=top_k)
            dt_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(dt_ms)

            pred = [meta.get("go_id", "").upper() for _, meta in hits if meta.get("go_id")]

            if not gold:
                if all(not gid.startswith("GO:") for gid in pred):
                    neg_non_go += 1
                continue

            inter = set(pred) & gold

            hit_scores.append(1.0 if inter else 0.0)
            recall_scores.append(len(inter) / len(gold))
            p_at_k_scores.append(len(inter) / max(1, min(top_k, len(pred))))
            ndcg_scores.append(ndcg_at_k(pred, gold, top_k))

            rr = 0.0
            for rank, gid in enumerate(pred[:top_k], start=1):
                if gid in gold:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

        if neg_cases:
            neg_non_go_rate_runs.append(neg_non_go / len(neg_cases))

    return {
        "hit_at_k": statistics.mean(hit_scores) if hit_scores else 0.0,
        "recall_at_k": statistics.mean(recall_scores) if recall_scores else 0.0,
        "mrr_at_k": statistics.mean(mrr_scores) if mrr_scores else 0.0,
        "ndcg_at_k": statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
        "precision_at_k": statistics.mean(p_at_k_scores) if p_at_k_scores else 0.0,
        "avg_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "p50_latency_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "neg_non_go_rate": statistics.mean(neg_non_go_rate_runs) if neg_non_go_rate_runs else 0.0,
        "positive_cases": len(pos_cases),
        "negative_cases": len(neg_cases),
        "total_cases": len(cases),
    }


def to_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="IC ablation benchmark for retrieval quality")
    parser.add_argument("--benchmark-file", type=str, default="data/benchmark_go_ner_200.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-json", type=str, default="data/ic_ablation_latest.json")
    parser.add_argument("--out-md", type=str, default="data/ic_ablation_table.md")
    args = parser.parse_args()

    cases = load_cases(Path(args.benchmark_file))
    full = evaluate_mode(cases, top_k=args.top_k, use_ic=False, repeats=args.repeats)
    ic = evaluate_mode(cases, top_k=args.top_k, use_ic=True, repeats=args.repeats)

    latency_delta = (ic["avg_latency_ms"] - full["avg_latency_ms"]) / max(full["avg_latency_ms"], 1e-9)

    out = {
        "benchmark_file": args.benchmark_file,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "full": full,
        "ic": ic,
        "delta_ic_minus_full": {
            "hit_at_k": ic["hit_at_k"] - full["hit_at_k"],
            "recall_at_k": ic["recall_at_k"] - full["recall_at_k"],
            "mrr_at_k": ic["mrr_at_k"] - full["mrr_at_k"],
            "ndcg_at_k": ic["ndcg_at_k"] - full["ndcg_at_k"],
            "precision_at_k": ic["precision_at_k"] - full["precision_at_k"],
            "avg_latency_ms": ic["avg_latency_ms"] - full["avg_latency_ms"],
            "avg_latency_ratio": latency_delta,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# IC 消融结果（可直接粘贴到答辩材料）",
        "",
        f"- 数据集：`{args.benchmark_file}`",
        f"- Top-K：`{args.top_k}`，重复：`{args.repeats}` 次（平均）",
        "",
        "| 方案 | Hit@K | Recall@K | MRR@K | nDCG@K | P@K | Avg Latency | P50 Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 全量索引 | {to_pct(full['hit_at_k'])} | {to_pct(full['recall_at_k'])} | {full['mrr_at_k']:.3f} | {full['ndcg_at_k']:.3f} | {to_pct(full['precision_at_k'])} | {full['avg_latency_ms']:.1f} ms | {full['p50_latency_ms']:.1f} ms |",
        f"| IC 两阶段 | {to_pct(ic['hit_at_k'])} | {to_pct(ic['recall_at_k'])} | {ic['mrr_at_k']:.3f} | {ic['ndcg_at_k']:.3f} | {to_pct(ic['precision_at_k'])} | {ic['avg_latency_ms']:.1f} ms | {ic['p50_latency_ms']:.1f} ms |",
        "",
        "**IC - 全量（Δ）**",
        "",
        f"- Hit@K: {(ic['hit_at_k'] - full['hit_at_k']) * 100:+.1f} pp",
        f"- Recall@K: {(ic['recall_at_k'] - full['recall_at_k']) * 100:+.1f} pp",
        f"- MRR@K: {ic['mrr_at_k'] - full['mrr_at_k']:+.3f}",
        f"- nDCG@K: {ic['ndcg_at_k'] - full['ndcg_at_k']:+.3f}",
        f"- P@K: {(ic['precision_at_k'] - full['precision_at_k']) * 100:+.1f} pp",
        f"- Avg latency: {ic['avg_latency_ms'] - full['avg_latency_ms']:+.1f} ms ({latency_delta * 100:+.1f}%)",
    ]

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n".join(md_lines))
    print(f"\n已写入: {out_json}")
    print(f"已写入: {out_md}")


if __name__ == "__main__":
    main()
