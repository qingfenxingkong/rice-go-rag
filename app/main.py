from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .index_builder import build_index
from .models import AnswerResponse, QuestionRequest, NERRequest, NERResponse, NERItem
from .rag import answer_question, stream_answer_generator
from .vector_store import get_vector_store, reset_vector_store
from .neo4j_client import Neo4jClient


app = FastAPI(
    title="Rice GO Knowledge Graph RAG API",
    description=(
        "基于 Neo4j 中的 GO 知识图谱 + 向量检索 + DeepSeek LLM 的智能问答后端。"
    ),
    version="0.1.0",
)

# 允许从本地文件或任意来源调用（方便使用本地 frontend.html）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    # 简单检查：尝试加载向量索引
    try:
        get_vector_store()
        vector_status = "ok"
    except Exception as e:  # noqa: BLE001
        vector_status = f"error: {e}"

    return {
        "status": "ok",
        "vector_index": vector_status,
        "neo4j_uri": settings.neo4j_uri,
        "model": settings.deepseek_model,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest) -> AnswerResponse:
    try:
        return answer_question(payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/admin/rebuild_index")
def admin_rebuild_index() -> dict:
    """手动触发从 Neo4j 重建向量索引（仅在可信环境下使用）。"""
    try:
        build_index()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    from . import vector_store as vs  # 延迟导入以避免循环
    vs._vector_store = None  # type: ignore[attr-defined]

    return {"status": "ok", "message": "FAISS 索引已重建。"}


@app.post("/admin/toggle_ic_index")
def admin_toggle_ic_index(enable: bool) -> dict:
    """动态切换是否使用 IC 压缩索引，无需重启服务。
    
    enable=true  → 使用 IC 压缩索引（更精准，术语更具代表性）
    enable=false → 恢复全量索引
    """
    from . import config as cfg
    cfg.settings.use_ic_index = enable
    reset_vector_store()  # 清空缓存，下次请求自动加载对应索引
    mode = "IC 压缩索引" if enable else "全量索引"
    return {"status": "ok", "use_ic_index": enable, "message": f"已切换至{mode}，下次检索生效。"}


@app.get("/admin/index_status")
def admin_index_status() -> dict:
    """查看当前使用的索引类型及加载状态。"""
    from . import vector_store as vs
    from .vector_store import TwoStageVectorStore
    from .config import settings as s
    store  = vs._vector_store  # type: ignore[attr-defined]
    loaded = store is not None
    if not loaded:
        size       = 0
        index_type = "IC 两阶段" if s.use_ic_index else "全量"
    elif isinstance(store, TwoStageVectorStore):
        size       = len(store.full_store.metadata)
        index_type = "IC 两阶段（IC粗筛 + 全量精筛）"
    else:
        size       = len(store.metadata)
        index_type = "全量索引"
    return {
        "use_ic_index":    s.use_ic_index,
        "index_type":      index_type,
        "loaded":          loaded,
        "term_count":      size,
        "ic_index_path":   s.ic_faiss_index_path,
        "full_index_path": s.faiss_index_path,
    }


@app.post("/ask_stream")
def ask_question_stream(payload: QuestionRequest):
    """流式返回答案，仅返回文本，不包含 sources 结构化信息。"""
    try:
        gen = stream_answer_generator(payload)
        return StreamingResponse(gen, media_type="text/plain; charset=utf-8")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ner", response_model=NERResponse)
def ner_endpoint(payload: NERRequest) -> NERResponse:
    """GO 术语实体识别接口。
    
    支持方法：dict / nltk / vector / llm / nltk_llm / nltk_token_llm / ensemble
    返回识别出的 GO 术语列表及相关信息。
    """
    import time
    import sys
    import os

    # 确保项目根目录在 sys.path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    OBO_PATH        = os.path.join(root, "go.obo")
    OLLAMA_BASE     = settings.deepseek_base_url if hasattr(settings, "deepseek_base_url") else "http://localhost:11434/v1"
    OLLAMA_MODEL    = settings.deepseek_model if hasattr(settings, "deepseek_model") else "deepseek-r1:14b"

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

    try:
        t0 = time.perf_counter()
        method = payload.method or "ensemble"
        mode   = payload.ensemble_mode or "balanced"
        text   = payload.text.strip()
        profile = settings.get_profile(payload.ner_profile, purpose="ner")

        if method == "dict":
            from go_ner.dict_ner import DictNER
            ner  = DictNER(obo_path=OBO_PATH, fuzzy_threshold=0.85, use_fuzzy=True)
            hits = ner.recognize(text)
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.score,
                             matched_text=r.span, source="dict")
                     for r in hits]

        elif method == "nltk":
            from go_ner.nltk_ner import NLTKNER
            ner  = NLTKNER(obo_path=OBO_PATH)
            hits = ner.recognize(text)
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.score,
                             matched_text=r.span, source="nltk")
                     for r in hits]

        elif method == "vector":
            from go_ner.vector_ner import VectorNER
            ner   = VectorNER(obo_path=OBO_PATH, model_path=profile["embedding_model_name"],
                              index_path=profile["faiss_index_path"], metadata_path=profile["faiss_metadata_path"], top_k=5)
            sents = ner.recognize_sentences(text, top_k=3)
            items = []
            for sent_hits in sents.values():
                for r in sent_hits:
                    if r.similarity >= 0.5:
                        items.append(NERItem(go_id=r.go_id, name=r.go_name or "",
                                             namespace=r.namespace, score=r.similarity,
                                             source="vector"))

        elif method == "llm":
            from go_ner.llm_ner import LLMNER
            ner  = LLMNER(obo_path=OBO_PATH, api_base=OLLAMA_BASE,
                          api_key="dummy", model=OLLAMA_MODEL, normalizer="dict",
                          conservative=True)
            hits = ner.recognize(text, lang=payload.lang or "auto")
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.confidence,
                             matched_text=r.original_span, source="llm")
                     for r in hits if r.go_id]

        elif method == "nltk_llm":
            from go_ner.nltk_llm_ner import NLTKLLMNER
            ner = NLTKLLMNER(
                obo_path=OBO_PATH,
                api_base=OLLAMA_BASE,
                api_key="dummy",
                model=OLLAMA_MODEL,
                llm_threshold=0.55,
            )
            hits = ner.recognize(text, lang=payload.lang or "auto")
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.score,
                             matched_text=r.span, source=r.source)
                     for r in hits]

        elif method == "nltk_token_llm":
            from go_ner.nltk_token_llm_ner import NLTKTokenLLMNER
            ner = NLTKTokenLLMNER(
                obo_path=OBO_PATH,
                api_base=OLLAMA_BASE,
                api_key="dummy",
                model=OLLAMA_MODEL,
                max_ngram=5,
                llm_threshold=0.55,
            )
            hits = ner.recognize(text, lang=payload.lang or "auto")
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.confidence,
                             matched_text=r.original_span, source="nltk_token_llm")
                     for r in hits if r.go_id]

        else:  # ensemble
            cfg = presets.get(mode, presets["balanced"])
            from go_ner.ensemble_ner import EnsembleNER
            ner  = EnsembleNER(
                obo_path=OBO_PATH, use_dict=True, use_vector=True, use_llm=True,
                model_path=profile["embedding_model_name"], index_path=profile["faiss_index_path"],
                metadata_path=profile["faiss_metadata_path"], vector_top_k=3,
                llm_api_base=OLLAMA_BASE, llm_api_key="dummy",
                llm_model=OLLAMA_MODEL, **cfg,
            )
            hits  = ner.recognize(text)
            items = [NERItem(go_id=r.go_id, name=r.go_name or "",
                             namespace=r.namespace, score=r.score,
                             matched_text=r.span, source="ensemble")
                     for r in hits]

        elapsed = time.perf_counter() - t0
        # 去重（同一 go_id 保留分数最高的）
        seen = {}
        for item in items:
            if item.go_id not in seen or (item.score or 0) > (seen[item.go_id].score or 0):
                seen[item.go_id] = item
        deduped = sorted(seen.values(), key=lambda x: x.score or 0, reverse=True)

        return NERResponse(text=text, method=method,
                           items=deduped, elapsed_s=round(elapsed, 3))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/graph_full")
def get_full_graph(limit_nodes: int = 200, limit_edges: int = 400) -> dict:
    """返回全图的一个可视化子图（类似 Neo4j Browser 的全局关系视图）。"""
    try:
        from .neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        nodes_map = {}
        edges = []
        try:
            with client._driver.session(database=settings.neo4j_database) as session:
                # 先取一批关系，关系两端节点自动带出
                res = session.run("""
                    MATCH (a)-[r]->(b)
                    WHERE (a:GO_Term OR a:Gene OR a:PMID OR a:RTO_Term)
                      AND (b:GO_Term OR b:Gene OR b:PMID OR b:RTO_Term)
                    RETURN a, b, type(r) AS rel
                    LIMIT $limit_edges
                """, limit_edges=limit_edges)

                for rec in res:
                    a = rec["a"]
                    b = rec["b"]
                    rel = rec["rel"]

                    def node_to_dict(n):
                        labels = list(n.labels)
                        nid = str(n.get("GO_Term") or n.get("RTO_Term") or n.get("EntrezID") or n.get("name") or id(n))
                        if "Gene" in labels:
                            nid = "Gene:" + str(n.get("EntrezID") or n.get("name") or nid)
                            ntype = "Gene"
                            label = str(n.get("name") or n.get("EntrezID") or "Gene")
                            ns = "Gene"
                        elif "PMID" in labels:
                            raw = str(n.get("name") or n.get("PMID") or nid)
                            nid = raw if raw.startswith("PMID:") else f"PMID:{raw}"
                            ntype = "PMID"
                            label = raw.replace("PMID:", "")
                            ns = "Literature"
                        elif "RTO_Term" in labels:
                            raw = str(n.get("RTO_Term") or n.get("name") or nid)
                            nid = raw if raw.startswith("RTO:") else f"RTO:{raw}"
                            ntype = "RTO_Term"
                            label = str(n.get("name") or raw)
                            ns = "RTO"
                        else:
                            raw = str(n.get("GO_Term") or nid)
                            nid = raw
                            ntype = "GO_Term"
                            label = str(n.get("name") or raw)
                            ns = str(n.get("Namespace") or "")
                        return {"id": nid, "label": label, "namespace": ns, "node_type": ntype}

                    na = node_to_dict(a)
                    nb = node_to_dict(b)
                    nodes_map[na["id"]] = na
                    nodes_map[nb["id"]] = nb
                    edges.append({"source": na["id"], "target": nb["id"], "relationship": rel})

                # 限制节点数量
                nodes = list(nodes_map.values())[:limit_nodes]
                node_ids = {n["id"] for n in nodes}
                edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

        finally:
            client.close()

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/graph")
def get_knowledge_graph(payload: QuestionRequest) -> dict:
    """根据问题检索相关 GO 术语，从 Neo4j 查询所有相关节点和关系并返回。"""
    try:
        from .rag import retrieve_sources, _expand_query_with_ner
        from .neo4j_client import get_neo4j_client

        question = payload.question
        retrieval_query = question
        if payload.use_ner:
            try:
                retrieval_query = _expand_query_with_ner(
                    question,
                    method=(payload.ner_method or "ensemble"),
                    ensemble_mode=(payload.ner_ensemble_mode or "balanced"),
                )
            except Exception:
                retrieval_query = question

        sources = retrieve_sources(retrieval_query)

        if not sources:
            return {"nodes": [], "edges": []}

        go_ids = [s.go_id for s in sources]
        print(f"[graph] 检索到 {len(go_ids)} 个 GO ID: {go_ids[:5]}")

        # 分离 GO_Term ID 和其他类型节点 ID
        pure_go_ids = [s.go_id for s in sources if s.node_type == "GO_Term" or s.go_id.startswith("GO:")]
        pmid_ids    = [s.go_id for s in sources if s.node_type == "PMID"   or s.go_id.startswith("PMID:")]
        gene_ids    = [s.go_id for s in sources if s.node_type == "Gene"   or s.go_id.startswith("Gene:")]
        print(f"[graph] GO_Term: {len(pure_go_ids)}, PMID: {len(pmid_ids)}, Gene: {len(gene_ids)}")

        # 同时准备大写和小写两种形式，兼容不同存储格式
        go_ids_all   = list(set(go_ids + [g.upper() for g in go_ids] + [g.lower() for g in go_ids]))

        client = get_neo4j_client()
        nodes_map = {}  # id -> node dict
        edges = []

        try:
            with client._driver.session(database=settings.neo4j_database) as session:

                # 1. GO_Term 节点（来自检索结果）
                for s in sources:
                    if s.node_type == "GO_Term" or s.go_id.startswith("GO:"):
                        nodes_map[s.go_id] = {
                            "id": s.go_id,
                            "label": s.name,
                            "namespace": s.namespace,
                            "node_type": "GO_Term",
                        }

                # 1b. 若检索结果都是 PMID/Gene，从它们反向找关联的 GO_Term
                if not nodes_map and pmid_ids:
                    pmid_vals = [p.replace("PMID:", "") for p in pmid_ids]
                    pmid_vals_prefixed = [f"PMID:{v}" for v in pmid_vals]
                    res = session.run("""
                        MATCH (go:GO_Term)-[:GO_Mention_in_Literature]->(p:PMID)
                        WHERE toString(p.name) IN $pmid_vals
                           OR toString(p.name) IN $pmid_vals_prefixed
                           OR toString(p.PMID) IN $pmid_vals
                           OR toString(p.PMID) IN $pmid_vals_prefixed
                        RETURN DISTINCT go.GO_Term AS go_id, go.name AS go_name,
                               go.Namespace AS ns
                        LIMIT 20
                    """, pmid_vals=pmid_vals, pmid_vals_prefixed=pmid_vals_prefixed)
                    for rec in res:
                        gid = rec["go_id"]
                        if gid and gid not in nodes_map:
                            nodes_map[gid] = {
                                "id": gid,
                                "label": rec["go_name"] or gid,
                                "namespace": rec["ns"] or "",
                                "node_type": "GO_Term",
                            }
                    # 更新 go_ids_all 以包含这些新找到的 GO ID
                    new_go_ids = list(nodes_map.keys())
                    go_ids_all.extend(new_go_ids)
                    go_ids_all = list(set(go_ids_all))
                    print(f"[graph] 从 PMID 反向找到 {len(new_go_ids)} 个 GO_Term")

                # 2. GO_Term 之间的关系（IS_A / RELATIONSHIP / part_of 等）
                res = session.run("""
                    MATCH (a:GO_Term)-[r]->(b:GO_Term)
                    WHERE a.GO_Term IN $go_ids OR b.GO_Term IN $go_ids
                    RETURN a.GO_Term AS src_id, a.name AS src_name, a.Namespace AS src_ns,
                           b.GO_Term AS tgt_id, b.name AS tgt_name, b.Namespace AS tgt_ns,
                           type(r) AS rel
                    LIMIT 80
                """, go_ids=go_ids_all)
                for rec in res:
                    src_id = rec["src_id"]
                    tgt_id = rec["tgt_id"]
                    if src_id and tgt_id:
                        if src_id not in nodes_map:
                            nodes_map[src_id] = {"id": src_id, "label": rec["src_name"] or src_id, "namespace": rec["src_ns"] or "", "node_type": "GO_Term"}
                        if tgt_id not in nodes_map:
                            nodes_map[tgt_id] = {"id": tgt_id, "label": rec["tgt_name"] or tgt_id, "namespace": rec["tgt_ns"] or "", "node_type": "GO_Term"}
                        edges.append({"source": src_id, "target": tgt_id, "relationship": rec["rel"]})

                # 2b. 若仍没有 GO-GO 边，尝试加入 1~2 跳的间接关系（提升可视化连通性）
                if not edges and nodes_map:
                    seed_ids = list(nodes_map.keys())
                    res = session.run("""
                        MATCH p=(a:GO_Term)-[*1..2]-(b:GO_Term)
                        WHERE a.GO_Term IN $seed_ids AND b.GO_Term IN $seed_ids
                        UNWIND relationships(p) AS r
                        WITH DISTINCT r
                        RETURN startNode(r).GO_Term AS src_id,
                               startNode(r).name AS src_name,
                               startNode(r).Namespace AS src_ns,
                               endNode(r).GO_Term AS tgt_id,
                               endNode(r).name AS tgt_name,
                               endNode(r).Namespace AS tgt_ns,
                               type(r) AS rel
                        LIMIT 120
                    """, seed_ids=seed_ids)
                    for rec in res:
                        src_id = rec["src_id"]
                        tgt_id = rec["tgt_id"]
                        if src_id and tgt_id:
                            if src_id not in nodes_map:
                                nodes_map[src_id] = {"id": src_id, "label": rec["src_name"] or src_id, "namespace": rec["src_ns"] or "", "node_type": "GO_Term"}
                            if tgt_id not in nodes_map:
                                nodes_map[tgt_id] = {"id": tgt_id, "label": rec["tgt_name"] or tgt_id, "namespace": rec["tgt_ns"] or "", "node_type": "GO_Term"}
                            edges.append({"source": src_id, "target": tgt_id, "relationship": rec["rel"]})
                    print(f"[graph] 间接关系补充后 edges={len(edges)}")

                # 3. Gene 节点及 Gene->GO_Term 关系（GO_Mention）
                res = session.run("""
                    MATCH (g:Gene)-[r:GO_Mention]->(go:GO_Term)
                    WHERE go.GO_Term IN $go_ids
                    RETURN g.EntrezID AS gene_id, g.name AS gene_name,
                           go.GO_Term AS go_id, type(r) AS rel
                    LIMIT 30
                """, go_ids=go_ids_all)
                for rec in res:
                    gene_id = "Gene:" + str(rec["gene_id"] or rec["gene_name"] or "unknown")
                    gene_label = str(rec["gene_name"] or rec["gene_id"] or "Gene")
                    go_id = rec["go_id"]
                    if gene_id not in nodes_map:
                        nodes_map[gene_id] = {"id": gene_id, "label": gene_label, "namespace": "Gene", "node_type": "Gene"}
                    if go_id and go_id in nodes_map:
                        edges.append({"source": gene_id, "target": go_id, "relationship": rec["rel"]})

                # 4. PMID 节点及相关关系（GO_Mention_in_Literature）
                res = session.run("""
                    MATCH (go:GO_Term)-[r:GO_Mention_in_Literature]->(p:PMID)
                    WHERE go.GO_Term IN $go_ids
                    RETURN go.GO_Term AS go_id,
                           p.name AS pmid, p.Title AS title, type(r) AS rel
                    LIMIT 20
                """, go_ids=go_ids_all)
                for rec in res:
                    pmid_val = str(rec["pmid"] or "PMID")
                    pmid_id = "PMID:" + pmid_val
                    pmid_label = pmid_val
                    go_id = rec["go_id"]
                    if pmid_id not in nodes_map:
                        nodes_map[pmid_id] = {"id": pmid_id, "label": pmid_label, "namespace": "Literature", "node_type": "PMID"}
                    if go_id in nodes_map:
                        edges.append({"source": go_id, "target": pmid_id, "relationship": rec["rel"]})

                # 5. RTO_Term 节点及相关关系
                res = session.run("""
                    MATCH (t:RTO_Term)-[r]->(go:GO_Term)
                    WHERE go.GO_Term IN $go_ids
                    RETURN t.RTO_Term AS rto_id, t.name AS rto_name,
                           go.GO_Term AS go_id, type(r) AS rel
                    LIMIT 20
                """, go_ids=go_ids_all)
                for rec in res:
                    rto_id = "RTO:" + str(rec["rto_id"] or rec["rto_name"] or "unknown")
                    rto_label = str(rec["rto_name"] or rec["rto_id"] or "RTO_Term")
                    go_id = rec["go_id"]
                    if rto_id not in nodes_map:
                        nodes_map[rto_id] = {"id": rto_id, "label": rto_label, "namespace": "RTO", "node_type": "RTO_Term"}
                    if go_id in nodes_map:
                        edges.append({"source": rto_id, "target": go_id, "relationship": rec["rel"]})

        finally:
            client.close()

        # 去重边（避免同一关系重复显示）
        uniq_edges = []
        seen_edges = set()
        for e in edges:
            key = (e.get("source"), e.get("target"), e.get("relationship"))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            uniq_edges.append(e)

        print(f"[graph] 返回 nodes={len(nodes_map)}, edges={len(uniq_edges)}")
        return {"nodes": list(nodes_map.values()), "edges": uniq_edges}

    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e

