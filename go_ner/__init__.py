"""
go_ner/__init__.py
GO 术语实体识别与概念标准化工具包

提供六种方案：
  1. DictNER      - 基于字典的精确/模糊匹配（快速，适合批量）
  2. NLTKNER      - 基于 NLTK 分词 + n-gram 字典匹配（传统 NLP baseline）
  3. VectorNER    - 基于向量语义相似度检索（适合非标准描述）
  4. LLMNER       - 基于大模型的 NER + 标准化（最智能）
  5. RAGLLMNER    - 基于检索增强候选约束的 LLM NER
  6. EnsembleNER  - 级联混合（速度+准确+召回平衡）
"""

from .dict_ner import DictNER, NERResult
from .nltk_ner import NLTKNER, NLTKNERResult
from .vector_ner import VectorNER, VectorNERResult
from .llm_ner import LLMNER, LLMNERResult
from .rag_llm_ner import RAGLLMNER
from .ensemble_ner import EnsembleNER, EnsembleNERResult
from .go_merge import GOMerger, MergeCandidate, MergedGO

__all__ = [
    "DictNER", "NERResult",
    "NLTKNER", "NLTKNERResult",
    "VectorNER", "VectorNERResult",
    "LLMNER", "LLMNERResult",
    "RAGLLMNER",
    "EnsembleNER", "EnsembleNERResult",
    "GOMerger", "MergeCandidate", "MergedGO",
]
