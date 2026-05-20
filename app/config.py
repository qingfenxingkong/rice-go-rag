import os
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


class Settings(BaseModel):
    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # DeepSeek / 通义等最终答案生成模型
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # 通用默认 embedding（兼容旧代码）
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "offline")  # offline 或 sbert
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Vector index（兼容旧代码）
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/go_faiss.index")
    faiss_metadata_path: str = os.getenv("FAISS_METADATA_PATH", "data/go_metadata.json")
    top_k: int = int(os.getenv("TOP_K", "5"))

    # 默认 profile：允许实体识别与 RAG 分开选择
    default_ner_profile: str = os.getenv("DEFAULT_NER_PROFILE", "pubmedbert")
    default_rag_profile: str = os.getenv("DEFAULT_RAG_PROFILE", "pubmedbert")

    # Profile 1: PubMedBERT / biomedical 专业模型
    pubmedbert_embedding_backend: str = os.getenv("PUBMEDBERT_EMBEDDING_BACKEND", "sbert")
    pubmedbert_embedding_model_name: str = os.getenv(
        "PUBMEDBERT_EMBEDDING_MODEL_NAME",
        os.getenv("EMBEDDING_MODEL_NAME", "models/S-PubMedBert-MS-MARCO"),
    )
    pubmedbert_faiss_index_path: str = os.getenv(
        "PUBMEDBERT_FAISS_INDEX_PATH",
        os.getenv("FAISS_INDEX_PATH", "data/go_faiss.index"),
    )
    pubmedbert_faiss_metadata_path: str = os.getenv(
        "PUBMEDBERT_FAISS_METADATA_PATH",
        os.getenv("FAISS_METADATA_PATH", "data/go_metadata.json"),
    )

    # Profile 2: MiniLM / 原来的通用模型
    minilm_embedding_backend: str = os.getenv("MINILM_EMBEDDING_BACKEND", "sbert")
    minilm_embedding_model_name: str = os.getenv(
        "MINILM_EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    minilm_faiss_index_path: str = os.getenv("MINILM_FAISS_INDEX_PATH", "data/go_faiss_minilm.index")
    minilm_faiss_metadata_path: str = os.getenv("MINILM_FAISS_METADATA_PATH", "data/go_metadata_minilm.json")

    # Profile 3: SapBERT / biomedical entity normalization 专用模型
    sapbert_embedding_backend: str = os.getenv("SAPBERT_EMBEDDING_BACKEND", "sbert")
    sapbert_embedding_model_name: str = os.getenv(
        "SAPBERT_EMBEDDING_MODEL_NAME",
        os.getenv("EMBEDDING_MODEL_NAME", "models/SapBERT-from-PubMedBERT-fulltext"),
    )
    sapbert_faiss_index_path: str = os.getenv("SAPBERT_FAISS_INDEX_PATH", "data/go_faiss_sapbert.index")
    sapbert_faiss_metadata_path: str = os.getenv("SAPBERT_FAISS_METADATA_PATH", "data/go_metadata_sapbert.json")

    # IC 压缩索引（可选）
    use_ic_index: bool = os.getenv("USE_IC_INDEX", "false").lower() == "true"
    ic_faiss_index_path: str = os.getenv("IC_FAISS_INDEX_PATH", "data/ic_index/go_ic_faiss.index")
    ic_faiss_metadata_path: str = os.getenv("IC_FAISS_METADATA_PATH", "data/ic_index/go_ic_metadata.json")

    def get_profile(self, profile: str | None, purpose: str = "rag") -> dict[str, str]:
        selected = (profile or "").strip().lower()
        if not selected:
            selected = self.default_ner_profile if purpose == "ner" else self.default_rag_profile

        if selected == "pubmedbert":
            return {
                "name": "pubmedbert",
                "embedding_backend": self.pubmedbert_embedding_backend,
                "embedding_model_name": self.pubmedbert_embedding_model_name,
                "faiss_index_path": self.pubmedbert_faiss_index_path,
                "faiss_metadata_path": self.pubmedbert_faiss_metadata_path,
            }

        if selected == "minilm":
            return {
                "name": "minilm",
                "embedding_backend": self.minilm_embedding_backend,
                "embedding_model_name": self.minilm_embedding_model_name,
                "faiss_index_path": self.minilm_faiss_index_path,
                "faiss_metadata_path": self.minilm_faiss_metadata_path,
            }

        if selected == "sapbert":
            return {
                "name": "sapbert",
                "embedding_backend": self.sapbert_embedding_backend,
                "embedding_model_name": self.sapbert_embedding_model_name,
                "faiss_index_path": self.sapbert_faiss_index_path,
                "faiss_metadata_path": self.sapbert_faiss_metadata_path,
            }

        raise ValueError(f"不支持的 profile: {selected}，可选值为 pubmedbert / minilm / sapbert")


settings = Settings()

