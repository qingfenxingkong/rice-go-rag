## 水稻 GO 知识图谱 + RAG 问答系统

本项目基于 Neo4j 中的 GO 术语图谱与向量检索，构建一个面向水稻分子机理的 RAG 问答系统。

### 功能模块

- **知识库构建（Neo4j）**：以 `GO_Term` 节点为核心实体，结合 `name`、`Namespace`、`Synonyms`、`Term_Description` 等属性构建知识图谱。
- **语义向量索引（FAISS）**：使用 Sentence Transformers 模型对 GO 文本描述向量化，存入 FAISS 高性能向量库，实现语义检索。
- **RAG 智能问答（FastAPI + DeepSeek）**：后端先通过向量检索召回相关 GO 术语，再将其作为上下文交给 DeepSeek 大模型生成答案。

### 目录结构

- `app/`
  - `config.py`：配置加载（Neo4j、DeepSeek、向量索引路径等）
  - `neo4j_client.py`：Neo4j 连接与 GO 术语查询
  - `models.py`：Pydantic / 数据模型
  - `embedding.py`：文本向量化（Sentence Transformer 或 DeepSeek Embedding 预留）
  - `index_builder.py`：从 Neo4j 导出 GO 术语并构建 FAISS 索引
  - `vector_store.py`：封装向量索引的加载与检索
  - `rag.py`：RAG 流程封装（检索 + 调用 DeepSeek 生成答案）
  - `main.py`：FastAPI 入口，提供 HTTP 接口
- `data/`
  - `go_faiss.index`：FAISS 索引文件（运行构建脚本后生成）
  - `go_metadata.json`：向量索引对应的 GO 元数据
- `database-all-data.csv`：原始 GO 注释数据（你已准备好，并导入 Neo4j）

### 环境配置

1. **创建虚拟环境并安装依赖**

```bash
cd 代码
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

2. **配置环境变量（推荐使用 `.env` 文件）**

在项目根目录新建 `.env`（可参考 `.env.example`）：

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat

EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
FAISS_INDEX_PATH=data/go_faiss.index
FAISS_METADATA_PATH=data/go_metadata.json
TOP_K=5
```

### 构建向量索引

启动前，需要先从 Neo4j 中拉取 GO 术语并构建向量索引：

```bash
python -m app.index_builder
```

执行成功后，会在 `data/` 目录下生成：

- `go_faiss.index`
- `go_metadata.json`

### 启动 FastAPI 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问接口示例：

- `GET /health`：健康检查
- `POST /ask`：RAG 问答

请求示例：

```bash
curl -X POST "http://localhost:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"水稻线粒体相关的遗传过程有哪些？\"}"
```

### 注意事项

- 本项目假设你已使用 `database-all-data.csv` 构建好了 Neo4j 图谱，且 `GO_Term` 节点至少包含以下属性：
  - `GO_Term`：GO ID（例如 `GO:0000001`）
  - `name`：术语标准名称
  - `Namespace`：本体命名空间（`biological_process` / `molecular_function` / `cellular_component`）
  - `Synonyms`：同义词（字符串，可能包含分号分隔）
  - `Term_Description`：详细描述
  - `Comment`：注释信息（如果有）
- 如果属性名与你的实际 Neo4j 图谱不完全一致，只需要在 `neo4j_client.py` 中调整对应字段名即可。

