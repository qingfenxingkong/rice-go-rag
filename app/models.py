from typing import List, Optional
from pydantic import BaseModel


class GOTerm(BaseModel):
    go_id: str
    name: str
    namespace: Optional[str] = None
    synonyms: Optional[str] = None
    description: Optional[str] = None
    comment: Optional[str] = None

    def build_text(self) -> str:
        parts: List[str] = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.go_id:
            parts.append(f"GO ID: {self.go_id}")
        if self.namespace:
            parts.append(f"Namespace: {self.namespace}")
        if self.synonyms:
            parts.append(f"Synonyms: {self.synonyms}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.comment:
            parts.append(f"Comment: {self.comment}")
        return "\n".join(parts)


class GeneTerm(BaseModel):
    gene_id: str
    name: str
    entrez_id: Optional[str] = None

    def build_text(self) -> str:
        parts: List[str] = []
        if self.name:
            parts.append(f"Gene Name: {self.name}")
        if self.entrez_id:
            parts.append(f"EntrezID: {self.entrez_id}")
        return "\n".join(parts)


class PMIDTerm(BaseModel):
    pmid: str
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None

    def build_text(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.journal:
            parts.append(f"Journal: {self.journal}")
        if self.year:
            parts.append(f"Year: {self.year}")
        return "\n".join(parts)


class RTOTerm(BaseModel):
    rto_id: str
    name: str
    description: Optional[str] = None

    def build_text(self) -> str:
        parts: List[str] = []
        if self.name:
            parts.append(f"RTO Term: {self.name}")
        if self.rto_id:
            parts.append(f"RTO ID: {self.rto_id}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_ner: Optional[bool] = False
    ner_method: Optional[str] = "ensemble"      # dict | nltk | vector | llm | nltk_llm | nltk_token_llm | ensemble
    ner_ensemble_mode: Optional[str] = "balanced"  # strict | balanced | recall
    rag_profile: Optional[str] = None            # pubmedbert | minilm | sapbert
    ner_profile: Optional[str] = None            # pubmedbert | minilm | sapbert


class SourceItem(BaseModel):
    go_id: str
    name: str
    score: float
    namespace: Optional[str] = None
    description: Optional[str] = None
    node_type: Optional[str] = "GO_Term"


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class NERRequest(BaseModel):
    text: str
    method: Optional[str] = "ensemble"   # dict | nltk | vector | llm | nltk_llm | nltk_token_llm | ensemble
    ensemble_mode: Optional[str] = "balanced"  # strict | balanced | recall
    lang: Optional[str] = "auto"
    ner_profile: Optional[str] = None     # pubmedbert | minilm | sapbert


class NERItem(BaseModel):
    go_id: str
    name: str
    namespace: Optional[str] = None
    score: Optional[float] = None
    matched_text: Optional[str] = None
    source: Optional[str] = None          # dict / nltk / vector / llm / ensemble


class NERResponse(BaseModel):
    text: str
    method: str
    items: List[NERItem]
    elapsed_s: float
