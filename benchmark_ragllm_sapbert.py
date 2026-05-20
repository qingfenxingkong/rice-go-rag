from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from go_ner.rag_llm_ner import RAGLLMNER


DEFAULT_CASES = "data/benchmark_go_ner_200.json"
DEFAULT_OUTPUT = "results/benchmark_ragllm_sapbert_200.json"
DEFAULT_MODEL_PATH = "models/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_INDEX_PATH = "data/go_faiss_sapbert.index"
DEFAULT_METADATA_PATH = "data/go_metadata_sapbert.json"
DEFAULT_LLM_API_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_LLM_MODEL = "deepseek-r1:14b"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" in payload:
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    raise ValueError("benchmark 文件格式不正确，应为 list 或 {'cases': [...]}。")


def _evaluate_case(case: dict[str, Any], ner: RAGLLMNER) -> dict[str, Any]:
    text = str(case.get("text", "")).strip()
    lang = str(case.get("lang", "auto") or "auto")
    gold = sorted(set(str(x).upper() for x in case.get("gold", [])))

    t0 = time.perf_counter()
    results = ner.recognize(text, lang=lang)
    elapsed = round(time.perf_counter() - t0, 3)

    pred = sorted(set(r.go_id.upper() for r in results if r.go_id))

    gold_set = set(gold)
    pred_set = set(pred)
    tp_ids = sorted(gold_set & pred_set)
    fp_ids = sorted(pred_set - gold_set)
    fn_ids = sorted(gold_set - pred_set)

    return {
        "id": case.get("id"),
        "text": text,
        "lang": lang,
        "difficulty": case.get("difficulty", ""),
        "gold": gold,
        "pred": pred,
        "tp_ids": tp_ids,
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "metrics": _compute_metrics(len(tp_ids), len(fp_ids), len(fn_ids)),
        "elapsed_s": elapsed,
        "results": [
            {
                "go_id": r.go_id,
                "go_name": r.go_name,
                "namespace": r.namespace,
                "definition": r.definition,
                "match_type": r.match_type,
                "confidence": round(float(r.confidence), 4),
                "span": r.original_span,
            }
            for r in results
        ],
    }


def _build_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(int(item["metrics"]["tp"]) for item in case_results)
    total_fp = sum(int(item["metrics"]["fp"]) for item in case_results)
    total_fn = sum(int(item["metrics"]["fn"]) for item in case_results)

    micro = _compute_metrics(total_tp, total_fp, total_fn)

    macro_precision = _safe_div(sum(float(item["metrics"]["precision"]) for item in case_results), len(case_results))
    macro_recall = _safe_div(sum(float(item["metrics"]["recall"]) for item in case_results), len(case_results))
    macro_f1 = _safe_div(sum(float(item["metrics"]["f1"]) for item in case_results), len(case_results))

    empty_pred_cases = sum(1 for item in case_results if not item["pred"])
    exact_match_cases = sum(1 for item in case_results if item["pred"] == item["gold"])
    avg_elapsed = _safe_div(sum(float(item["elapsed_s"]) for item in case_results), len(case_results))

    return {
        "case_count": len(case_results),
        "micro": micro,
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "exact_match_rate": round(_safe_div(exact_match_cases, len(case_results)), 4),
        "empty_prediction_rate": round(_safe_div(empty_pred_cases, len(case_results)), 4),
        "avg_elapsed_s": round(avg_elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 SapBERT 索引 + RAGLLMNER 跑 200 条 GO NER benchmark")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="benchmark JSON 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="结果 JSON 输出路径")
    parser.add_argument("--obo", default="go.obo", help="go.obo 路径")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="SapBERT 模型名或本地路径")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="SapBERT 索引路径")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="SapBERT 元数据路径")
    parser.add_argument("--llm-api-base", default=DEFAULT_LLM_API_BASE, help="LLM API 地址")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="LLM 模型名")
    parser.add_argument("--candidate-top-k", type=int, default=12, help="RAG 候选数量")
    parser.add_argument("--vector-threshold", type=float, default=0.70, help="向量候选阈值（提高以减少误报）")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_path = Path(args.output)

    if not Path(args.index_path).exists():
        raise FileNotFoundError(f"SapBERT 索引不存在: {args.index_path}")
    if not Path(args.metadata_path).exists():
        raise FileNotFoundError(f"SapBERT 元数据不存在: {args.metadata_path}")

    cases = _load_cases(cases_path)

    ner = RAGLLMNER(
        obo_path=args.obo,
        model_path=args.model_path,
        api_base=args.llm_api_base,
        api_key="dummy",
        model=args.llm_model,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        candidate_top_k=args.candidate_top_k,
        vector_threshold=args.vector_threshold,
        conservative=True,
    )

    case_results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] case_id={case.get('id')} lang={case.get('lang', 'auto')}")
        try:
            case_results.append(_evaluate_case(case, ner))
        except Exception as ex:
            case_results.append({
                "id": case.get("id"),
                "text": case.get("text", ""),
                "lang": case.get("lang", "auto"),
                "difficulty": case.get("difficulty", ""),
                "gold": case.get("gold", []),
                "error": str(ex),
            })
            print(f"  -> failed: {ex}")
            continue

        metrics = case_results[-1]["metrics"]
        print(
            f"  -> pred={len(case_results[-1]['pred'])} "
            f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
            f"f1={metrics['f1']:.4f}"
        )

    valid_results = [item for item in case_results if "error" not in item]
    summary = _build_summary(valid_results)

    payload = {
        "benchmark_file": str(cases_path),
        "obo": args.obo,
        "model_path": args.model_path,
        "index_path": args.index_path,
        "metadata_path": args.metadata_path,
        "llm_api_base": args.llm_api_base,
        "llm_model": args.llm_model,
        "candidate_top_k": args.candidate_top_k,
        "vector_threshold": args.vector_threshold,
        "summary": summary,
        "results": case_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已保存 -> {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
