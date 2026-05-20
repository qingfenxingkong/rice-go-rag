from __future__ import annotations

from typing import List

from neo4j import GraphDatabase, Driver

from .config import settings
from .models import GOTerm, GeneTerm, PMIDTerm, RTOTerm


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def _get_session(self):
        return self._driver.session(database=self._database)

    def fetch_all_go_terms(self) -> List[GOTerm]:
        query = """
        MATCH (g:GO_Term)
        RETURN
            g.GO_Term AS go_id,
            g.name AS name,
            g.Namespace AS namespace,
            g.Synonyms AS synonyms,
            g.Term_Description AS description,
            g.Comment AS comment
        ORDER BY go_id
        """
        results: List[GOTerm] = []
        with self._get_session() as session:
            for record in session.run(query):
                go_id = record.get("go_id")
                name = record.get("name")
                if not go_id or not name:
                    continue
                namespace = record.get("namespace")
                synonyms = record.get("synonyms")
                description = record.get("description")
                comment = record.get("comment")
                results.append(
                    GOTerm(
                        go_id=str(go_id),
                        name=str(name),
                        namespace=str(namespace) if namespace not in (None, "") else None,
                        synonyms=str(synonyms) if synonyms not in (None, "") else None,
                        description=str(description) if description not in (None, "") else None,
                        comment=str(comment) if comment not in (None, "") else None,
                    )
                )
        return results

    def fetch_all_genes(self) -> List[GeneTerm]:
        query = """
        MATCH (g:Gene)
        RETURN g.name AS name, g.EntrezID AS entrez_id
        """
        results: List[GeneTerm] = []
        with self._get_session() as session:
            for record in session.run(query):
                name = record.get("name")
                entrez_id = record.get("entrez_id")
                if not name:
                    continue
                gene_id = "Gene:" + str(entrez_id or name)
                results.append(GeneTerm(
                    gene_id=gene_id,
                    name=str(name),
                    entrez_id=str(entrez_id) if entrez_id else None,
                ))
        return results

    def fetch_all_pmids(self) -> List[PMIDTerm]:
        query = """
        MATCH (p:PMID)
        RETURN p.name AS pmid, p.Title AS title, p.Journal AS journal, p.Year AS year
        """
        results: List[PMIDTerm] = []
        with self._get_session() as session:
            for record in session.run(query):
                pmid = record.get("pmid")
                title = record.get("title")
                if not pmid and not title:
                    continue
                results.append(PMIDTerm(
                    pmid="PMID:" + str(pmid or ""),
                    title=str(title) if title else None,
                    journal=str(record.get("journal")) if record.get("journal") else None,
                    year=str(record.get("year")) if record.get("year") else None,
                ))
        return results

    def fetch_all_rto_terms(self) -> List[RTOTerm]:
        query = """
        MATCH (t:RTO_Term)
        RETURN t.RTO_Term AS rto_id, t.name AS name, t.Term_Description AS description
        """
        results: List[RTOTerm] = []
        with self._get_session() as session:
            for record in session.run(query):
                rto_id = record.get("rto_id")
                name = record.get("name")
                if not rto_id and not name:
                    continue
                description = record.get("description")
                results.append(RTOTerm(
                    rto_id="RTO:" + str(rto_id or name),
                    name=str(name or rto_id),
                    description=str(description) if description not in (None, "") else None,
                ))
        return results


def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
