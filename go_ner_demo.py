"""
go_ner_demo.py
GO NER - demo script

Usage:
  python go_ner_demo.py --method dict
  python go_ner_demo.py --method vector
  python go_ner_demo.py --method llm
  python go_ner_demo.py --method ensemble
  python go_ner_demo.py --method benchmark
  python go_ner_demo.py --method benchmark --benchmark-file data/benchmark_go_ner_200.json
  python go_ner_demo.py --method hierarchy
  python go_ner_demo.py --method dict --text "your text here"

  # IC ablation (separate script):
  python benchmark_ic.py
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OBO_PATH        = "go.obo"
EMBEDDING_MODEL = r"C:\Users\yyq\Desktop\毕业设计\代码\paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX     = "data/go_faiss.index"
FAISS_META      = "data/go_metadata.json"
OLLAMA_BASE     = "http://localhost:11434/v1"
OLLAMA_MODEL    = "deepseek-r1:14b"

SAMPLE_TEXTS = [
    (
        "Rice plants regulate mitochondrion inheritance during cell division. "
        "The process involves DNA repair mechanisms and kinase activity in the nucleus."
    ),
    "GO:0006950 (response to stress) and GO:0009409 were upregulated in drought conditions.",
    "\u6c34\u7a3b\u7684\u7ebf\u7c92\u4f53\u9057\u4f20\u4e0e\u7ec6\u80de\u5206\u88c2\u8fc7\u7a0b\u5bc6\u5207\u76f8\u5173\uff0cDNA\u635f\u4f24\u4fee\u590d\u548c\u6fc0\u9178\u6d3b\u6027\u5728\u5176\u4e2d\u53d1\u6325\u91cd\u8981\u4f5c\u7528\u3002",
    (
        "The gene product localizes to the inner membrane of the powerhouse organelle "
        "and participates in energy production via oxidative phosphorylation pathway."
    ),
]

BENCHMARK_CASES = [
    {
        "text": (
            "Rice plants regulate mitochondrion inheritance during cell division, "
            "and DNA repair supports genome stability."
        ),
        "gold": {"GO:0000001", "GO:0051301", "GO:0006281"},
    },
    {
        "text": "Kinase activity in the nucleus modulates signal transduction under stress response.",
        "gold": {"GO:0016301", "GO:0005634", "GO:0006950"},
    },
    {
        "text": "Oxidative phosphorylation and ATP synthesis occur in mitochondrion inner membrane.",
        "gold": {"GO:0006119", "GO:0005743"},
    },
    {
        "text": "\u690d\u7269\u7ec6\u80de\u5206\u88c2\u548cDNA\u4fee\u590d\u8fc7\u7a0b\u53d7\u6fc0\u9178\u6d3b\u6027\u8c03\u63a7\u3002",
        "gold": {"GO:0051301", "GO:0006281", "GO:0016301"},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _calc_prf1(tp: int, fp: int, fn: int):
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _evaluate_predictions(pred_ids, gold_ids):
    tp = len(pred_ids & gold_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)
    return tp, fp, fn


def _load_benchmark_cases(benchmark_file: str) -> list:
    if not benchmark_file:
        return BENCHMARK_CASES
    try:
        with open(benchmark_file, encoding="utf-8") as f:
            obj = json.load(f)
        cases = obj.get("cases", []) if isinstance(obj, dict) else []
        normalized = []
        for c in cases:
            txt = str(c.get("text", "")).strip()
            if not txt:
                continue
            gold = c.get("gold", [])
            if isinstance(gold, list):
                gold_set = {str(x).upper() for x in gold if str(x).startswith("GO:")}
            else:
                gold_set = set()
            normalized.append({
                "id":         c.get("id", len(normalized) + 1),
                "text":       txt,
                "gold":       gold_set,
                "lang":       c.get("lang", "unknown"),
                "difficulty": c.get("difficulty", "unknown"),
            })
        if normalized:
            print(f"[Benchmark] Loaded {len(normalized)} cases from {benchmark_file}")
            return normalized
    except Exception as ex:
        print(f"[Benchmark] Failed to load ({ex}), falling back to built-in cases")
    return BENCHMARK_CASES


def _is_negative_case(case: dict) -> bool:
    g = case.get("gold", set())
    return len(g) == 0


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
def run_hierarchy_compress() -> None:
    print("\n" + "=" * 70)
    print("GO Hierarchy Compression + IC + DictNER")
    print("=" * 70)

    from go_ner.go_hierarchy import GOHierarchy
    from go_ner.dict_ner import DictNER
    import os
    from dotenv import load_dotenv
    load_dotenv()

    hier = GOHierarchy(OBO_PATH)
    hier.print_stats()

    ic_struct = hier.compute_ic_structural()
    for ns in ["biological_process", "molecular_function", "cellular_component"]:
        print(f"\n[{ns}]")
        nodes = hier.select_representative_nodes_by_ic(
            ic=ic_struct, namespace=ns,
            min_depth=3, max_depth=6,
            min_descendants=20, max_nodes=200,
            coverage_threshold=0.90,
            ic_min=2.0, ic_max=10.0,
        )
        print(f"  Compressed: {len(nodes)} representative nodes")
        top5 = sorted(nodes, key=lambda n: ic_struct.get(n.go_id, 0), reverse=True)[:5]
        for n in top5:
            print(f"    [{n.go_id}] IC={ic_struct.get(n.go_id, 0):.2f} "
                  f"depth={n.depth} desc={n.descendant_count:4d}  {n.name}")

    neo4j_uri      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
    neo4j_user     = os.getenv("NEO4J_USER",     "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    ic_neo4j, ann_counts = hier.compute_ic_from_neo4j(
        neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
        neo4j_password=neo4j_password, neo4j_database=neo4j_database,
    )
    compressed = hier.get_compressed_dict_by_ic(
        use_neo4j=True,
        neo4j_uri=neo4j_uri, neo4j_user=neo4j_user,
        neo4j_password=neo4j_password, neo4j_database=neo4j_database,
        min_depth=2, max_depth=8, min_descendants=0, max_nodes=2000,
        coverage_threshold=0.95, ic_min=3.0, ic_max=14.0, min_annotations=0,
    )
    print(f"\n  Full ~42036 -> compressed {len(compressed)} terms")
    ner = DictNER(obo_path=OBO_PATH, entries=compressed, use_fuzzy=True)
    text = SAMPLE_TEXTS[0]
    print(f"\n  Test text: {text}")
    results = ner.recognize(text)
    print(ner.format_results(text, results))

    ic_struct2 = hier.compute_ic_structural()
    sample_ids = ["GO:0006950", "GO:0006281", "GO:0016301", "GO:0008150", "GO:0000001"]
    print(f"  {'GO ID':<15} {'Ann':^6} {'Name':<40} {'Struct IC':^10} {'Neo4j IC':^10}")
    print("  " + "-" * 83)
    for gid in sample_ids:
        node = hier.get_node(gid)
        if node:
            ann = ann_counts.get(gid, 0)
            print(f"  {gid:<15} {ann:^6d} {node.name[:38]:<40} "
                  f"{ic_struct2.get(gid, 0):^10.3f} {ic_neo4j.get(gid, 0):^10.3f}")


def run_dict_method(text: str) -> None:
    print("\n" + "=" * 70)
    print("Method 1: Dictionary NER (DictNER)")
    print("=" * 70)
    from go_ner.dict_ner import DictNER
    ner = DictNER(obo_path=OBO_PATH, fuzzy_threshold=0.85, use_fuzzy=True)
    results = ner.recognize(text)
    print(ner.format_results(text, results))
    print("\n-- Normalization examples --")
    for term in ["mitochondrion inheritance", "DNA repair",
                 "GO:0000001", "kinase activity"]:
        entry = ner.normalize(term)
        if entry:
            print(f"  '{term}' -> {entry.go_id} | {entry.name} ({entry.namespace})")
        else:
            print(f"  '{term}' -> not found")


def run_vector_method(text: str) -> None:
    print("\n" + "=" * 70)
    print("Method 2: Vector Semantic NER (VectorNER)")
    print("=" * 70)
    from go_ner.vector_ner import VectorNER
    ner = VectorNER(
        obo_path=OBO_PATH,
        model_path=EMBEDDING_MODEL,
        index_path=FAISS_INDEX,
        metadata_path=FAISS_META,
        top_k=5,
    )
    print("\n-- Full text search Top-5 --")
    results = ner.search(text, top_k=5)
    for r in results:
        print(f"  [{r.similarity:.4f}] {r.go_id} | {r.go_name} ({r.namespace})")
        print(f"           def: {r.definition[:80]}...")
    print("\n-- Per-sentence search Top-3 --")
    sent_results = ner.recognize_sentences(text, top_k=3)
    print(ner.format_results(sent_results, threshold=0.4))
    print("\n-- Normalization examples --")
    for term in ["powerhouse organelle inner membrane", "energy production",
                 "oxidative phosphorylation"]:
        r = ner.normalize(term)
        if r:
            print(f"  '{term}'")
            print(f"    -> [{r.similarity:.4f}] {r.go_id} | {r.go_name}")


def run_llm_method(text: str) -> None:
    print("\n" + "=" * 70)
    print("Method 3: LLM NER + Dict Normalization (LLMNER)")
    print("=" * 70)
    from go_ner.llm_ner import LLMNER
    ner = LLMNER(
        obo_path=OBO_PATH,
        api_base=OLLAMA_BASE,
        api_key="dummy",
        model=OLLAMA_MODEL,
        normalizer="dict",
    )
    results = ner.recognize(text, lang="auto")
    print(ner.format_results(results))


def run_ensemble_method(text: str, mode: str = "balanced") -> None:
    print("\n" + "=" * 70)
    print(f"Method 4: Ensemble NER (mode={mode})")
    print("=" * 70)
    presets = {
        "strict":   dict(vector_threshold=0.72, llm_threshold=0.65, weight_dict=0.70,
                         weight_vector=0.20, weight_llm=0.10, consensus_bonus=0.06,
                         final_threshold=0.80, min_consensus=1),
        "balanced": dict(vector_threshold=0.60, llm_threshold=0.50, weight_dict=0.60,
                         weight_vector=0.25, weight_llm=0.15, consensus_bonus=0.05,
                         final_threshold=0.70, min_consensus=1),
        "recall":   dict(vector_threshold=0.50, llm_threshold=0.40, weight_dict=0.50,
                         weight_vector=0.30, weight_llm=0.20, consensus_bonus=0.03,
                         final_threshold=0.58, min_consensus=1),
    }
    cfg = presets.get(mode, presets["balanced"])
    from go_ner.ensemble_ner import EnsembleNER
    ner = EnsembleNER(
        obo_path=OBO_PATH,
        use_dict=True, use_vector=True, use_llm=True,
        model_path=EMBEDDING_MODEL,
        index_path=FAISS_INDEX,
        metadata_path=FAISS_META,
        vector_top_k=3,
        llm_api_base=OLLAMA_BASE,
        llm_api_key="dummy",
        llm_model=OLLAMA_MODEL,
        **cfg,
    )
    results = ner.recognize(text)
    print(ner.format_results(text, results))


def run_all(text: str, ensemble_mode: str = "balanced") -> None:
    run_dict_method(text)
    run_vector_method(text)
    run_llm_method(text)
    run_ensemble_method(text, mode=ensemble_mode)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(ensemble_mode: str = "balanced",
                  benchmark_file: str = "data/benchmark_go_ner_200.json") -> None:
    print("\n" + "=" * 70)
    print(f"GO NER Benchmark (ensemble_mode={ensemble_mode})")
    print("=" * 70)

    cases = _load_benchmark_cases(benchmark_file)

    from go_ner.dict_ner     import DictNER
    from go_ner.vector_ner   import VectorNER
    from go_ner.llm_ner      import LLMNER
    from go_ner.ensemble_ner import EnsembleNER

    presets = {
        "strict":   dict(vector_threshold=0.72, llm_threshold=0.65, weight_dict=0.70,
                         weight_vector=0.20, weight_llm=0.10, consensus_bonus=0.06,
                         final_threshold=0.80, min_consensus=1),
        "balanced": dict(vector_threshold=0.60, llm_threshold=0.50, weight_dict=0.60,
                         weight_vector=0.25, weight_llm=0.15, consensus_bonus=0.05,
                         final_threshold=0.70, min_consensus=1),
        "recall":   dict(vector_threshold=0.50, llm_threshold=0.40, weight_dict=0.50,
                         weight_vector=0.30, weight_llm=0.20, consensus_bonus=0.03,
                         final_threshold=0.58, min_consensus=1),
    }
    cfg = presets.get(ensemble_mode, presets["balanced"])

    dict_ner   = DictNER(obo_path=OBO_PATH, fuzzy_threshold=0.85, use_fuzzy=True)
    vector_ner = VectorNER(obo_path=OBO_PATH, model_path=EMBEDDING_MODEL,
                           index_path=FAISS_INDEX, metadata_path=FAISS_META, top_k=5)
    llm_ner    = LLMNER(obo_path=OBO_PATH, api_base=OLLAMA_BASE,
                        api_key="dummy", model=OLLAMA_MODEL, normalizer="dict")
    ens_ner    = EnsembleNER(
        obo_path=OBO_PATH, use_dict=True, use_vector=True, use_llm=True,
        model_path=EMBEDDING_MODEL, index_path=FAISS_INDEX, metadata_path=FAISS_META,
        vector_top_k=3, llm_api_base=OLLAMA_BASE, llm_api_key="dummy",
        llm_model=OLLAMA_MODEL, **cfg,
    )

    methods = {
        "dict":                    dict_ner,
        "vector":                  vector_ner,
        "llm":                     llm_ner,
        f"ensemble-{ensemble_mode}": ens_ner,
    }

    stats = {m: {"tp": 0, "fp": 0, "fn": 0, "t": 0.0, "neg_fp": 0, "neg_n": 0}
             for m in methods}
    grp_lang = {m: {} for m in methods}
    grp_diff = {m: {} for m in methods}

    def _acc(bucket, m, key, tp, fp, fn):
        if key not in bucket[m]:
            bucket[m][key] = {"tp": 0, "fp": 0, "fn": 0}
        bucket[m][key]["tp"] += tp
        bucket[m][key]["fp"] += fp
        bucket[m][key]["fn"] += fn

    def _predict(m_name, ner_obj, text):
        t0 = time.perf_counter()
        if "vector" in m_name:
            sents = ner_obj.recognize_sentences(text, top_k=1)
            pred = {hits[0].go_id for hits in sents.values()
                    if hits and hits[0].similarity >= 0.5}
        elif "llm" in m_name:
            pred = {r.go_id for r in ner_obj.recognize(text, lang="auto") if r.go_id}
        elif "ensemble" in m_name:
            pred = {r.go_id for r in ner_obj.recognize(text)}
        else:
            pred = {r.go_id for r in ner_obj.recognize(text)}
        return pred, time.perf_counter() - t0

    n = len(cases)
    for idx, case in enumerate(cases, 1):
        text = case["text"]
        gold = case["gold"]
        is_neg = len(gold) == 0
        lang   = case.get("lang", "unknown")
        diff   = case.get("difficulty", "unknown")
        short  = text[:60] + "..." if len(text) > 60 else text
        print(f"  [{idx:3d}/{n}] {short}")
        for m_name, ner_obj in methods.items():
            pred, dt = _predict(m_name, ner_obj, text)
            tp = len(pred & gold)
            fp = len(pred - gold)
            fn = len(gold - pred)
            s = stats[m_name]
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn; s["t"] += dt
            _acc(grp_lang, m_name, lang, tp, fp, fn)
            _acc(grp_diff, m_name, diff, tp, fp, fn)
            if is_neg:
                s["neg_n"] += 1
                if pred:
                    s["neg_fp"] += 1

    neg_total = sum(1 for c in cases if len(c["gold"]) == 0)
    print(f"\nBenchmark cases: {n}  (negatives: {neg_total})")
    print(f"{'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AvgTime(s)':>12} {'Neg-FP%':>10}")
    print("-" * 78)
    for m in methods:
        s = stats[m]
        p, r, f1 = _calc_prf1(s["tp"], s["fp"], s["fn"])
        avg_t = s["t"] / n
        neg_rate = (s["neg_fp"] / s["neg_n"] * 100.0) if s["neg_n"] else 0.0
        print(f"{m:<22} {p:>10.3f} {r:>10.3f} {f1:>10.3f} {avg_t:>12.3f} {neg_rate:>9.1f}%")

    print("\n" + "-" * 78)
    print("F1 by language")
    print("-" * 78)
    lang_keys = sorted({k for m in methods for k in grp_lang[m]})
    print(f"{'Method':<22}" + "".join(f"{k:>12}" for k in lang_keys))
    for m in methods:
        row = f"{m:<22}"
        for k in lang_keys:
            g = grp_lang[m].get(k)
            row += f"{_calc_prf1(g['tp'],g['fp'],g['fn'])[2]:>12.3f}" if g else f"{'−':>12}"
        print(row)

    print("\n" + "-" * 78)
    print("F1 by difficulty")
    print("-" * 78)
    diff_keys = sorted({k for m in methods for k in grp_diff[m]})
    print(f"{'Method':<22}" + "".join(f"{k:>16}" for k in diff_keys))
    for m in methods:
        row = f"{m:<22}"
        for k in diff_keys:
            g = grp_diff[m].get(k)
            row += f"{_calc_prf1(g['tp'],g['fp'],g['fn'])[2]:>16.3f}" if g else f"{'−':>16}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GO NER demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python go_ner_demo.py --method dict
          python go_ner_demo.py --method vector
          python go_ner_demo.py --method llm
          python go_ner_demo.py --method ensemble --ensemble-mode strict
          python go_ner_demo.py --method benchmark --benchmark-file data/benchmark_go_ner_200.json
          python benchmark_ic.py   # IC ablation study
        """)
    )
    parser.add_argument(
        "--method",
        choices=["dict", "vector", "llm", "ensemble", "hierarchy", "benchmark", "all"],
        default="dict",
    )
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--ensemble-mode",
        choices=["strict", "balanced", "recall"],
        default="balanced",
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="data/benchmark_go_ner_200.json",
    )
    args = parser.parse_args()

    text = args.text or SAMPLE_TEXTS[args.sample % len(SAMPLE_TEXTS)]

    if args.method == "hierarchy":
        run_hierarchy_compress()
    elif args.method == "benchmark":
        run_benchmark(ensemble_mode=args.ensemble_mode,
                      benchmark_file=args.benchmark_file)
    else:
        print(f"\nInput text:\n{text}\n")
        if args.method == "dict":
            run_dict_method(text)
        elif args.method == "vector":
            run_vector_method(text)
        elif args.method == "llm":
            run_llm_method(text)
        elif args.method == "ensemble":
            run_ensemble_method(text, mode=args.ensemble_mode)
        elif args.method == "all":
            run_all(text, ensemble_mode=args.ensemble_mode)


if __name__ == "__main__":
    main()
        