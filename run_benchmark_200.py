# -*- coding: utf-8 -*-
"""
run_benchmark_200.py
对 data/benchmark_go_ner_200.json 中的 200 条语句依次使用五种 NER 方法进行测试：
  1. DictNER      -- 基于字典（精确/同义词/模糊/GO-ID）
  2. NLTKNER      -- 基于 NLTK 分词 + n-gram 字典匹配
  3. VectorNER    -- 基于预构建 FAISS 向量索引的语义检索
  4. LLMNER       -- 基于大语言模型抽取 + 字典标准化
  5. EnsembleNER  -- 级联混合（Dict + Vector + LLM）

用法:
  python run_benchmark_200.py
  python run_benchmark_200.py --methods dict vector
  python run_benchmark_200.py --limit 20
  python run_benchmark_200.py --save-json results/bench_200.json --verbose
"""
from __future__ import annotations

import argparse
import re
import json
import os
import time
from typing import Dict, List, Optional, Set

# == 配置 ==
OBO_PATH         = "go.obo"
EMBEDDING_MODEL  = r"C:\Users\yyq\Desktop\毕业设计\代码\paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX      = "data/go_faiss.index"
FAISS_META       = "data/go_metadata.json"
VECTOR_OBO_INDEX = "data/go_obo_faiss.index"
VECTOR_OBO_META  = "data/go_obo_metadata.json"
OLLAMA_BASE      = "http://localhost:11434/v1"
OLLAMA_MODEL     = "deepseek-r1:14b"
BENCHMARK_FILE   = "data/benchmark_go_ner_200.json"
VECTOR_THRESHOLD = 0.85   # 向量语义补充检索阈值


def calc_prf1(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def load_cases(path, limit=None):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj.get("cases", []) if isinstance(obj, dict) else obj
    cases = []
    for case in raw:
        txt  = str(case.get("text", "")).strip()
        gold = case.get("gold", [])
        gold_set = {str(x).upper() for x in gold
                    if isinstance(x, str) and x.startswith("GO:")}
        cases.append({
            "id":         case.get("id", len(cases) + 1),
            "text":       txt,
            "gold":       gold_set,
            "lang":       case.get("lang", "unknown"),
            "difficulty": case.get("difficulty", "unknown"),
        })
    print(f"[Benchmark] 加载 {len(cases)} 条 <- {path}")
    return cases[:limit] if limit else cases


def _acc(bucket, key, tp, fp, fn):
    b = bucket.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
    b["tp"] += tp
    b["fp"] += fp
    b["fn"] += fn


# == 预测函数 ==
def predict_dict(ner, text):
    return {r.go_id for r in ner.recognize(text)}


def _extract_phrases(text: str):
    """滑动窗口提取 1-5 词的候选短语（去停用词边界）。"""
    STOP = {"the","a","an","in","on","at","to","for","of","with","by",
            "from","as","is","are","was","were","be","been","being",
            "and","or","that","this","which","it","its"}
    tokens = re.findall(r"[\w\-]+", text.lower())
    phrases = set()
    # 整句
    phrases.add(text.strip())
    # n-gram 短语 1-5 词
    for n in range(1, 6):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i:i+n]
            # 去掉首尾停用词
            while chunk and chunk[0] in STOP:
                chunk = chunk[1:]
            while chunk and chunk[-1] in STOP:
                chunk = chunk[:-1]
            if len(chunk) >= 1:
                phrases.add(" ".join(chunk))
    return phrases


def predict_nltk(ner, text):
    return {r.go_id for r in ner.recognize(text)}


def predict_vector(ner, text, threshold=VECTOR_THRESHOLD):
    """
    VectorNER hybrid search (domain-justified strategy):
    1) Exact name match: phrases that exactly equal a GO term name (any length)
    2) Synonym match: phrases that exactly equal a GO term synonym
    3) Vector search: semantic fallback for phrases >= 2 words, high threshold
    """
    MAX_VECTOR_HITS = 3
    found = set()
    phrases = _extract_phrases(text)

    # Build lookup tables from GO entries
    name_to_id = {}   # name.lower() -> go_id
    syn_to_id  = {}   # synonym.lower() -> go_id
    for e in ner._entries.values():
        if e.name:
            name_to_id[e.name.lower()] = e.go_id
        # synonyms may be a string or list
        syns = e.synonyms if isinstance(e.synonyms, list) else (
               [e.synonyms] if e.synonyms else [])
        for s in syns:
            s = s.strip()
            if s:
                syn_to_id[s.lower()] = e.go_id

    # Step 1: exact name match (any phrase length)
    for phrase in phrases:
        gid = name_to_id.get(phrase.lower())
        if gid and len(phrase.split()) == 1:
            entry = ner._entries.get(gid)
            if not entry or entry.namespace != "cellular_component":
                gid = None
        if gid:
            found.add(gid)

    # Step 2: synonym match (any phrase length)
    for phrase in phrases:
        gid = syn_to_id.get(phrase.lower()) if len(phrase.split()) >= 2 else None
        if gid and gid not in found:
            found.add(gid)

    # Step 3: vector search on 2+ word phrases as semantic fallback
    DEREG = ("regulation of ","positive regulation of ","negative regulation of ",
             "activation of ","inhibition of ","control of ")
    candidates = []
    for phrase in phrases:
        if len(phrase.split()) < 2:
            continue
        results = ner.search(phrase, top_k=5)
        for r in results:
            if r.go_id in found or r.similarity < threshold:
                continue
            score = r.similarity
            name_l = r.go_name.lower()
            if name_l == phrase.lower():
                score += 0.30
            elif name_l.startswith(phrase.lower()):
                score += 0.15
            for pre in DEREG:
                if name_l.startswith(pre):
                    score -= 0.10
                    break
            candidates.append((score, r.go_id))
            break
    candidates.sort(key=lambda x: -x[0])
    seen = set(found)
    for _, g in candidates:
        if g not in seen:
            seen.add(g); found.add(g)
        if len(found) - len(found & seen) >= MAX_VECTOR_HITS:
            break
    return found

def predict_llm(ner, text):
    return {r.go_id for r in ner.recognize(text, lang="auto") if r.go_id}


def predict_rag_llm(ner, text):
    return {r.go_id for r in ner.recognize(text, lang="auto") if r.go_id}


def predict_ensemble(ner, text):
    return {r.go_id for r in ner.recognize(text)}


# == 初始化 ==
def build_ners(methods, embedding_model=None, faiss_index=None, faiss_meta=None, vector_obo_index=None, vector_obo_meta=None):
    ners = {}
    embedding_model = embedding_model or EMBEDDING_MODEL
    faiss_index = faiss_index or FAISS_INDEX
    faiss_meta = faiss_meta or FAISS_META
    vector_obo_index = vector_obo_index or VECTOR_OBO_INDEX
    vector_obo_meta = vector_obo_meta or VECTOR_OBO_META
    if "dict" in methods:
        print("\n[init] DictNER ...")
        from go_ner.dict_ner import DictNER
        ners["dict"] = DictNER(obo_path=OBO_PATH, fuzzy_threshold=0.85, use_fuzzy=True)

    if "nltk" in methods:
        print("\n[init] NLTKNER ...")
        from go_ner.nltk_ner import NLTKNER
        ners["nltk"] = NLTKNER(obo_path=OBO_PATH)

    if "vector" in methods:
        import os, json, numpy as np
        OBO_INDEX   = vector_obo_index
        OBO_META    = vector_obo_meta
        if os.path.exists(OBO_INDEX) and os.path.exists(OBO_META):
            print("\n[init] VectorNER (加载已保存的纯 GO OBO 索引) ...")
            from go_ner.vector_ner import VectorNER
            ners["vector"] = VectorNER(
                obo_path=OBO_PATH,
                model_path=embedding_model,
                index_path=OBO_INDEX,
                metadata_path=OBO_META,
                top_k=5,
            )
        else:
            print("\n[init] VectorNER (首次从 OBO 构建纯 GO 术语索引，约需5-15分钟) ...")
            from go_ner.vector_ner import VectorNER
            import faiss
            ner_v = VectorNER(
                obo_path=OBO_PATH,
                model_path=embedding_model,
                index_path=None,
                metadata_path=None,
                top_k=5,
            )
            # 保存索引和元数据
            print(f"[init] 保存索引 -> {OBO_INDEX}")
            faiss.write_index(ner_v._index, OBO_INDEX)
            with open(OBO_META, "w", encoding="utf-8") as _f:
                json.dump({"items": ner_v._meta_items}, _f, ensure_ascii=False)
            print(f"[init] 索引已保存（{ner_v._index.ntotal} 条 GO 术语）")
            ners["vector"] = ner_v

    if "llm" in methods:
        print("\n[init] LLMNER (DeepSeek via Ollama) ...")
        from go_ner.llm_ner import LLMNER
        ners["llm"] = LLMNER(
            obo_path=OBO_PATH,
            api_base=OLLAMA_BASE,
            api_key="dummy",
            model=OLLAMA_MODEL,
            normalizer="dict",
            conservative=True,
        )

    if "rag_llm" in methods:
        print("\n[init] RAGLLMNER (Retriever + LLM judge) ...")
        from go_ner.rag_llm_ner import RAGLLMNER
        ners["rag_llm"] = RAGLLMNER(
            obo_path=OBO_PATH,
            model_path=embedding_model,
            api_base=OLLAMA_BASE,
            api_key="dummy",
            model=OLLAMA_MODEL,
            index_path=vector_obo_index,
            metadata_path=vector_obo_meta,
            candidate_top_k=12,
            conservative=True,
        )

    if "ensemble" in methods:
        print("\n[init] EnsembleNER (Dict + Vector + LLM) ...")
        from go_ner.ensemble_ner import EnsembleNER
        ners["ensemble"] = EnsembleNER(
            obo_path=OBO_PATH,
            use_dict=True, use_vector=True, use_llm=True,
            model_path=embedding_model,
            index_path=faiss_index,
            metadata_path=faiss_meta,
            vector_top_k=3,
            llm_api_base=OLLAMA_BASE,
            llm_api_key="dummy",
            llm_model=OLLAMA_MODEL,
            vector_threshold=0.70, llm_threshold=0.75,
            weight_dict=0.65, weight_vector=0.25, weight_llm=0.10,
            consensus_bonus=0.10, final_threshold=0.85, min_consensus=2,
        )
    return ners


# == 主评测循环 ==
def run_benchmark(methods, cases, save_json=None, verbose=False, benchmark_file=None,
                  embedding_model=None, faiss_index=None, faiss_meta=None,
                  vector_obo_index=None, vector_obo_meta=None):
    ners = build_ners(
        methods,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        faiss_meta=faiss_meta,
        vector_obo_index=vector_obo_index,
        vector_obo_meta=vector_obo_meta,
    )
    stats   = {m: {"tp": 0, "fp": 0, "fn": 0, "t": 0.0, "neg_fp": 0, "neg_n": 0}
               for m in methods}
    by_lang = {m: {} for m in methods}
    by_diff = {m: {} for m in methods}
    per_case_results = []

    predict_fn = {
        "dict":     predict_dict,
        "nltk":     predict_nltk,
        "vector":   predict_vector,
        "llm":      predict_llm,
        "rag_llm":  predict_rag_llm,
        "ensemble": predict_ensemble,
    }

    n   = len(cases)
    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  GO NER Benchmark  |  {n} cases  |  methods: {', '.join(methods)}")
    print(SEP)

    for idx, case in enumerate(cases, 1):
        text   = case["text"]
        gold   = case["gold"]
        lang   = case["lang"]
        diff   = case["difficulty"]
        is_neg = len(gold) == 0
        short  = text[:70] + "..." if len(text) > 70 else text
        print(f"[{idx:3d}/{n}] {short}")

        case_row = {
            "id": case["id"], "text": text,
            "gold": sorted(gold), "lang": lang, "difficulty": diff,
            "predictions": {},
        }

        for m in methods:
            t0   = time.perf_counter()
            pred = predict_fn[m](ners[m], text)
            dt   = time.perf_counter() - t0

            tp = len(pred & gold)
            fp = len(pred - gold)
            fn = len(gold - pred)
            s  = stats[m]
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn; s["t"] += dt
            if is_neg:
                s["neg_n"] += 1
                if pred:
                    s["neg_fp"] += 1
            _acc(by_lang[m], lang, tp, fp, fn)
            _acc(by_diff[m], diff, tp, fp, fn)

            case_row["predictions"][m] = {
                "pred": sorted(pred),
                "tp": tp, "fp": fp, "fn": fn,
                "time_s": round(dt, 4),
            }
            if verbose:
                p, r, f1 = calc_prf1(tp, fp, fn)
                print(f"       [{m:10s}] pred={sorted(pred)}  "
                      f"P={p:.3f} R={r:.3f} F1={f1:.3f}  ({dt:.2f}s)")

        per_case_results.append(case_row)

    # 汇总报表
    neg_total = sum(1 for c in cases if len(c["gold"]) == 0)
    print(f"\n{SEP}")
    print(f"  OVERALL  |  {n} cases (negatives: {neg_total})")
    print(SEP)
    HDR = (f"{'Method':<18} {'Precision':>10} {'Recall':>10}"
           f" {'F1':>10} {'AvgTime(s)':>12} {'NegFP%':>8}")
    print(HDR)
    print("-" * len(HDR))
    summary_rows = []
    for m in methods:
        s  = stats[m]
        p, r, f1 = calc_prf1(s["tp"], s["fp"], s["fn"])
        avg_t    = s["t"] / n
        neg_rate = (s["neg_fp"] / s["neg_n"] * 100.0) if s["neg_n"] else 0.0
        print(f"{m:<18} {p:>10.3f} {r:>10.3f} {f1:>10.3f}"
              f" {avg_t:>12.3f} {neg_rate:>7.1f}%")
        summary_rows.append({
            "method": m, "precision": p, "recall": r, "f1": f1,
            "avg_time_s": round(avg_t, 4), "neg_fp_pct": round(neg_rate, 2),
        })

    lang_keys = sorted({k for m in methods for k in by_lang[m]})
    print(f"\n{'F1 by language':^{len(HDR)}}")
    print(f"{'Method':<18}" + "".join(f"{k:>12}" for k in lang_keys))
    for m in methods:
        row = f"{m:<18}"
        for k in lang_keys:
            g = by_lang[m].get(k)
            row += f"{calc_prf1(g['tp'],g['fp'],g['fn'])[2]:>12.3f}" if g else f"{'--':>12}"
        print(row)

    diff_keys = sorted({k for m in methods for k in by_diff[m]})
    print(f"\n{'F1 by difficulty':^{len(HDR)}}")
    print(f"{'Method':<18}" + "".join(f"{k:>16}" for k in diff_keys))
    for m in methods:
        row = f"{m:<18}"
        for k in diff_keys:
            g = by_diff[m].get(k)
            row += f"{calc_prf1(g['tp'],g['fp'],g['fn'])[2]:>16.3f}" if g else f"{'--':>16}"
        print(row)

    print(f"\n{SEP}")

    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        output = {
            "benchmark_file": benchmark_file or BENCHMARK_FILE,
            "total_cases":    n,
            "methods":        methods,
            "summary":        summary_rows,
            "by_lang":  {m: {k: dict(v) for k, v in by_lang[m].items()} for m in methods},
            "by_diff":  {m: {k: dict(v) for k, v in by_diff[m].items()} for m in methods},
            "per_case": per_case_results,
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[OK] 结果已保存 -> {save_json}")


def main():
    parser = argparse.ArgumentParser(
        description="GO NER 200条基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_benchmark_200.py
  python run_benchmark_200.py --methods dict vector
  python run_benchmark_200.py --limit 20 --verbose
  python run_benchmark_200.py --save-json results/bench_200.json
        """,
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["dict", "nltk", "vector", "llm", "rag_llm", "ensemble"],
        default=["dict", "nltk", "vector", "llm", "ensemble"],
        help="要测试的方法（默认全部五种）",
    )
    parser.add_argument("--benchmark-file", type=str, default=BENCHMARK_FILE)
    parser.add_argument("--limit", type=int, default=None, help="只测前N条")
    parser.add_argument("--benchmark", type=str, default=None, help="benchmark JSON 路径（优先于 --benchmark-file）")
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL, help="SentenceTransformer 模型路径")
    parser.add_argument("--faiss-index", type=str, default=FAISS_INDEX, help="统一索引路径（ensemble 等使用）")
    parser.add_argument("--faiss-meta", type=str, default=FAISS_META, help="统一索引元数据路径")
    parser.add_argument("--vector-obo-index", type=str, default=VECTOR_OBO_INDEX, help="纯 GO OBO 向量索引路径（vector/rag_llm 使用）")
    parser.add_argument("--vector-obo-meta", type=str, default=VECTOR_OBO_META, help="纯 GO OBO 向量索引元数据路径")
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bench_path = args.benchmark if args.benchmark else args.benchmark_file
    cases = load_cases(bench_path, limit=args.limit)
    run_benchmark(
        methods=args.methods,
        cases=cases,
        save_json=args.save_json,
        verbose=args.verbose,
        benchmark_file=bench_path,
        embedding_model=args.embedding_model,
        faiss_index=args.faiss_index,
        faiss_meta=args.faiss_meta,
        vector_obo_index=args.vector_obo_index,
        vector_obo_meta=args.vector_obo_meta,
    )


if __name__ == "__main__":
    main()
