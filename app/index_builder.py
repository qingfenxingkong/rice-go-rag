from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from .config import settings
from .embedding import encode_texts
from .models import GOTerm, GeneTerm, PMIDTerm, RTOTerm
from .neo4j_client import get_neo4j_client


def _collect_records_from_neo4j() -> tuple[List[GOTerm], List[GeneTerm], List[PMIDTerm], List[RTOTerm]]:
    client = get_neo4j_client()
    try:
        print("正在从 Neo4j 读取 GO_Term 节点...")
        go_terms = client.fetch_all_go_terms()
        print(f"  GO_Term: {len(go_terms)} 条")

        print("正在从 Neo4j 读取 Gene 节点...")
        genes = client.fetch_all_genes()
        print(f"  Gene: {len(genes)} 条")

        print("正在从 Neo4j 读取 PMID 节点...")
        pmids = client.fetch_all_pmids()
        print(f"  PMID: {len(pmids)} 条")

        print("正在从 Neo4j 读取 RTO_Term 节点...")
        rto_terms = client.fetch_all_rto_terms()
        print(f"  RTO_Term: {len(rto_terms)} 条")
        return go_terms, genes, pmids, rto_terms
    finally:
        client.close()


def _load_records_from_metadata() -> tuple[List[GOTerm], List[GeneTerm], List[PMIDTerm], List[RTOTerm]]:
    metadata_path = Path(settings.faiss_metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"离线重建需要已有元数据文件，但未找到: {metadata_path}"
        )

    print(f"Neo4j 不可用，改为从已有元数据离线重建: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", {})
    if not items:
        raise RuntimeError("已有元数据文件中没有 items，无法离线重建索引。")

    go_terms: List[GOTerm] = []
    genes: List[GeneTerm] = []
    pmids: List[PMIDTerm] = []
    rto_terms: List[RTOTerm] = []

    for item in items.values():
        node_type = item.get("node_type", "GO_Term")
        if node_type == "GO_Term":
            go_id = item.get("go_id")
            name = item.get("name")
            if not go_id or not name:
                continue
            go_terms.append(GOTerm(
                go_id=str(go_id),
                name=str(name),
                namespace=str(item.get("namespace")) if item.get("namespace") not in (None, "") else None,
                synonyms=str(item.get("synonyms")) if item.get("synonyms") not in (None, "") else None,
                description=str(item.get("description")) if item.get("description") not in (None, "") else None,
                comment=str(item.get("comment")) if item.get("comment") not in (None, "") else None,
            ))
        elif node_type == "Gene":
            gene_id = item.get("go_id")
            name = item.get("name")
            if not gene_id or not name:
                continue
            description = str(item.get("description") or "")
            entrez_id = None
            prefix = "EntrezID:"
            if description.startswith(prefix):
                raw = description[len(prefix):].strip()
                entrez_id = raw or None
            genes.append(GeneTerm(
                gene_id=str(gene_id),
                name=str(name),
                entrez_id=entrez_id,
            ))
        elif node_type == "PMID":
            pmid = item.get("go_id")
            name = item.get("name")
            if not pmid and not name:
                continue
            description = str(item.get("description") or "")
            journal = None
            year = None
            for part in description.split(","):
                part = part.strip()
                if part.startswith("Journal:"):
                    journal = part[len("Journal:"):].strip() or None
                elif part.startswith("Year:"):
                    year = part[len("Year:"):].strip() or None
            pmids.append(PMIDTerm(
                pmid=str(pmid or name or ""),
                title=str(name) if name not in (None, "") else None,
                journal=journal,
                year=year,
            ))
        elif node_type == "RTO_Term":
            rto_id = item.get("go_id")
            name = item.get("name")
            if not rto_id and not name:
                continue
            rto_terms.append(RTOTerm(
                rto_id=str(rto_id or name),
                name=str(name or rto_id),
                description=str(item.get("description")) if item.get("description") not in (None, "") else None,
            ))

    print(
        f"  离线元数据载入完成：GO_Term {len(go_terms)} 条, Gene {len(genes)} 条, "
        f"PMID {len(pmids)} 条, RTO_Term {len(rto_terms)} 条"
    )
    return go_terms, genes, pmids, rto_terms


def build_index() -> None:
    """优先从 Neo4j 读取所有节点构建统一的 FAISS 索引；若 Neo4j 不可用，则基于已有元数据离线重建。"""
    try:
        go_terms, genes, pmids, rto_terms = _collect_records_from_neo4j()
    except Exception as exc:
        print(f"Neo4j 读取失败：{exc}")
        go_terms, genes, pmids, rto_terms = _load_records_from_metadata()

    # 统一构建文本和元数据列表
    texts: List[str] = []
    metadata_list: List[Dict] = []

    for term in go_terms:
        texts.append(term.build_text())
        metadata_list.append({
            "node_type": "GO_Term",
            "go_id": term.go_id,
            "name": term.name,
            "namespace": term.namespace or "",
            "synonyms": term.synonyms or "",
            "description": term.description or "",
            "comment": term.comment or "",
        })

    for gene in genes:
        texts.append(gene.build_text())
        metadata_list.append({
            "node_type": "Gene",
            "go_id": gene.gene_id,
            "name": gene.name,
            "namespace": "Gene",
            "synonyms": "",
            "description": f"EntrezID: {gene.entrez_id or ''}",
            "comment": "",
        })

    for pmid in pmids:
        texts.append(pmid.build_text())
        metadata_list.append({
            "node_type": "PMID",
            "go_id": pmid.pmid,
            "name": pmid.title or pmid.pmid,
            "namespace": "Literature",
            "synonyms": "",
            "description": f"Journal: {pmid.journal or ''}, Year: {pmid.year or ''}",
            "comment": "",
        })

    for rto in rto_terms:
        texts.append(rto.build_text())
        metadata_list.append({
            "node_type": "RTO_Term",
            "go_id": rto.rto_id,
            "name": rto.name,
            "namespace": "RTO",
            "synonyms": "",
            "description": rto.description or "",
            "comment": "",
        })

    total = len(texts)
    if total == 0:
        raise RuntimeError("没有获取到任何节点数据，请检查 Neo4j 图谱。")

    print(f"\n共 {total} 条数据，开始向量化...")
    embeddings = encode_texts(texts)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings 必须是二维数组。")

    num_items, dim = embeddings.shape
    print(f"向量化完成：num_items={num_items}, dim={dim}，构建 FAISS 索引...")

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.faiss_metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"保存 FAISS 索引到: {index_path}")
    faiss.write_index(index, str(index_path))

    metadata: Dict[int, Dict] = {i: m for i, m in enumerate(metadata_list)}
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"dim": dim, "items": metadata}, f, ensure_ascii=False, indent=2)

    print(f"\n索引构建完成！共 {total} 条（GO_Term: {len(go_terms)}, Gene: {len(genes)}, PMID: {len(pmids)}, RTO_Term: {len(rto_terms)}）")


if __name__ == "__main__":
    build_index()
