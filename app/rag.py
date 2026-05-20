from __future__ import annotations

import re
from typing import Generator, List, Tuple

import requests

from .config import settings
from .models import AnswerResponse, QuestionRequest, SourceItem
from .vector_store import get_vector_store


def retrieve_sources(question: str, top_k: int | None = None, rag_profile: str | None = None) -> List[SourceItem]:
    """根据问题从向量索引中检索相关节点（GO_Term / Gene / PMID / RTO_Term）。"""
    if top_k is None:
        top_k = settings.top_k

    vector_store = get_vector_store(profile=rag_profile)
    results = vector_store.search(question, top_k=top_k)

    sources = []
    for score, metadata in results:
        source = SourceItem(
            go_id=metadata.get("go_id", ""),
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            description=metadata.get("description", ""),
            score=float(score),
            node_type=metadata.get("node_type", "GO_Term"),
        )
        sources.append(source)

    return sources


def _expand_query_with_ner(question: str, method: str, ensemble_mode: str, ner_profile: str | None = None) -> str:
    """可选：使用 NER 结果增强检索查询文本。"""
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    profile = settings.get_profile(ner_profile, purpose="ner")

    OBO_PATH = os.path.join(root, "go.obo")
    EMBEDDING_MODEL = profile["embedding_model_name"]
    FAISS_INDEX = profile["faiss_index_path"]
    FAISS_META = profile["faiss_metadata_path"]

    ner_terms: list[str] = []

    if method == "dict":
        from go_ner.dict_ner import DictNER
        ner = DictNER(obo_path=OBO_PATH, fuzzy_threshold=0.85, use_fuzzy=True)
        hits = ner.recognize(question)
        ner_terms = [f"{h.go_id} {h.go_name or ''}".strip() for h in hits]
    elif method == "nltk":
        from go_ner.nltk_ner import NLTKNER
        ner = NLTKNER(obo_path=OBO_PATH)
        hits = ner.recognize(question)
        ner_terms = [f"{h.go_id} {h.go_name or ''}".strip() for h in hits]
    elif method == "vector":
        from go_ner.vector_ner import VectorNER
        ner = VectorNER(
            obo_path=OBO_PATH,
            model_path=EMBEDDING_MODEL,
            index_path=FAISS_INDEX,
            metadata_path=FAISS_META,
            top_k=5,
        )
        sents = ner.recognize_sentences(question, top_k=3)
        for sent_hits in sents.values():
            for h in sent_hits:
                if h.similarity >= 0.5:
                    ner_terms.append(f"{h.go_id} {h.go_name or ''}".strip())
    elif method == "llm":
        from go_ner.llm_ner import LLMNER
        ner = LLMNER(
            obo_path=OBO_PATH,
            api_base=settings.deepseek_base_url,
            api_key="dummy",
            model=settings.deepseek_model,
            normalizer="dict",
            conservative=True,
        )
        hits = ner.recognize(question, lang="auto")
        ner_terms = [f"{h.go_id} {h.go_name or ''}".strip() for h in hits if h.go_id]
    else:
        from go_ner.ensemble_ner import EnsembleNER
        presets = {
            "strict": dict(vector_threshold=0.72, llm_threshold=0.65, weight_dict=0.70, weight_vector=0.20, weight_llm=0.10, consensus_bonus=0.06, final_threshold=0.80, min_consensus=1),
            "balanced": dict(vector_threshold=0.60, llm_threshold=0.50, weight_dict=0.60, weight_vector=0.25, weight_llm=0.15, consensus_bonus=0.05, final_threshold=0.70, min_consensus=1),
            "recall": dict(vector_threshold=0.50, llm_threshold=0.40, weight_dict=0.50, weight_vector=0.30, weight_llm=0.20, consensus_bonus=0.03, final_threshold=0.58, min_consensus=1),
        }
        cfg = presets.get(ensemble_mode, presets["balanced"])
        ner = EnsembleNER(
            obo_path=OBO_PATH,
            use_dict=True,
            use_vector=True,
            use_llm=True,
            model_path=EMBEDDING_MODEL,
            index_path=FAISS_INDEX,
            metadata_path=FAISS_META,
            vector_top_k=3,
            llm_api_base=settings.deepseek_base_url,
            llm_api_key="dummy",
            llm_model=settings.deepseek_model,
            **cfg,
        )
        hits = ner.recognize(question)
        ner_terms = [f"{h.go_id} {h.go_name or ''}".strip() for h in hits if h.go_id]

    uniq_terms = list(dict.fromkeys(t for t in ner_terms if t))[:6]
    if not uniq_terms:
        return question
    return f"{question}\nGO_HINTS: {' ; '.join(uniq_terms)}"


def build_context_from_sources(
    sources: List[Tuple[float, dict]],
) -> str:
    """将检索到的节点构造成提示词上下文，供模型理解其生物学含义。"""
    lines: List[str] = []
    lines.append(
        "The following are entries from a rice molecular biology knowledge graph "
        "including Gene Ontology (GO) terms, genes, literature references, and RTO traits. "
        "Use them strictly as factual background.\n"
    )
    for score, meta in sources:
        node_type = meta.get("node_type", "GO_Term")
        name = meta.get("name", "")
        description = meta.get("description", "")
        namespace = meta.get("namespace", "")

        if node_type == "GO_Term":
            lines.append(
                f"- [GO Term] Namespace: {namespace}\n"
                f"  Description: {description}\n"
            )
        elif node_type == "Gene":
            lines.append(
                f"- [Gene] Name: {name}\n"
                f"  {description}\n"
            )
        elif node_type == "PMID":
            lines.append(
                f"- [Literature] Title: {name}\n"
                f"  {description}\n"
            )
        elif node_type == "RTO_Term":
            lines.append(
                f"- [RTO Trait] Name: {name}\n"
                f"  Description: {description}\n"
            )
        else:
            lines.append(
                f"- [{node_type}] Name: {name}\n"
                f"  Description: {description}\n"
            )
    return "\n".join(lines)


_GO_ID_RE = re.compile(r"\bGO:\d{7}\b", re.IGNORECASE)
_GO_LABEL_RE = re.compile(r"\bGO\s*ID\b", re.IGNORECASE)


def _sanitize_answer_text(text: str) -> str:
    """确保最终回答不包含 GO ID 等标签信息。"""
    text = _GO_LABEL_RE.sub("", text)
    text = _GO_ID_RE.sub("", text)
    return text


def call_deepseek_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 DeepSeek Chat Completions API。

    DeepSeek API 与 OpenAI 接口基本兼容，这里使用 /chat/completions 端点。
    """
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请在环境变量或 .env 中设置。")

    url = f"{settings.deepseek_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    # 本地 Ollama 首次加载大模型可能较慢，这里给足超时时间
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"DeepSeek API 返回格式异常: {data}")

    return content


def stream_deepseek_llm(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    """以流式方式调用 DeepSeek / Ollama 的 Chat Completions 接口。

    返回一个生成器，逐块产出新增的文本内容，仅支持 OpenAI 风格的流式响应。
    """
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请在环境变量或 .env 中设置。")

    url = f"{settings.deepseek_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": True,
    }

    with requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180,
        stream=True,
    ) as resp:
        resp.raise_for_status()

        in_think = False

        # 以字节形式读取，再强制按 UTF-8 解码，避免编码被错误推断导致中文乱码
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue
            # 兼容 "data: {...}" 或直接就是 JSON 的两种情况
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                import json

                data = json.loads(line)
            except Exception:
                continue

            try:
                delta = data["choices"][0]["delta"]
                chunk = delta.get("content")
            except Exception:
                chunk = None

            if not chunk:
                continue

            # DeepSeek-R1 模型会在 <think>...</think> 中输出推理过程，这里做简单过滤：
            # 仅向前端流式输出思考标签之外的内容，避免出现大量“乱码”式的思维链。
            text = chunk
            out_chars: list[str] = []
            i = 0
            while i < len(text):
                if not in_think and text.startswith("<think>", i):
                    in_think = True
                    i += len("<think>")
                elif in_think and text.startswith("</think>", i):
                    in_think = False
                    i += len("</think>")
                else:
                    if not in_think:
                        out_chars.append(text[i])
                    i += 1

            filtered = "".join(out_chars)
            if filtered:
                yield filtered


def answer_question(payload: QuestionRequest) -> AnswerResponse:
    vector_store = get_vector_store(profile=payload.rag_profile)
    top_k = payload.top_k or settings.top_k

    query_for_retrieval = payload.question
    if payload.use_ner:
        try:
            query_for_retrieval = _expand_query_with_ner(
                payload.question,
                method=(payload.ner_method or "ensemble"),
                ensemble_mode=(payload.ner_ensemble_mode or "balanced"),
                ner_profile=payload.ner_profile,
            )
        except Exception:
            query_for_retrieval = payload.question

    hits = vector_store.search(query_for_retrieval, top_k=top_k)
    if not hits:
        # 没有召回任何内容时，仍然回答，但明确提示“基于一般知识”
        system_prompt = (
            "你是一名面向科研人员和公众的水稻分子机理知识助手。"
            "在回答时要："
            "1）用准确、理性的中文回答；"
            "2）尽量解释专业术语，保证非本领域研究者也能理解；"
            "3）如果依据不足，请明确说明“不确定”或“现有资料不足”，不要编造事实或夸大结论。"
        )
        user_prompt = (
            "当前没有检索到与下述问题直接关联的 GO 术语上下文，"
            "请基于你的一般生物学知识，尽量给出客观、保守的回答。\n\n"
            f"用户问题：{payload.question}\n\n"
            "回答结构：先用 1-2 句话给出总体结论，然后用条目列出 2-4 点补充说明。"
        )
        raw_answer = call_deepseek_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return AnswerResponse(answer=raw_answer, sources=[])

    context = build_context_from_sources(hits)

    system_prompt = (
        "你是一名面向科研人员和公众的水稻分子机理知识助手。"
        "只允许基于提供的 GO 术语及其描述进行推理和回答，"
        "禁止引入上下文之外的具体结论或未经支持的假设。"
        "在回答时要："
        "1）使用清晰、理性的中文；"
        "2）重点解释这些 GO 术语所代表的生物学过程或分子功能本身，而不是机械地重复术语的英文名称或原始描述；"
        "3）可以在括号中简要标出 GO ID 和一个简短中文名，但主体内容应是你用自己的话对其功能的专业、通俗解释；"
        "4）当依据不足时，明确说明“不确定”或“现有资料不足”。"
    )

    user_prompt = (
        "下面是与水稻分子机理相关的一组 GO 术语及其描述，请严格以此作为事实依据回答用户问题。"
        "请先从生物学机理的角度抽象出这些术语所代表的核心含义，再据此作答。\n\n"
        f"{context}\n\n"
        f"用户问题：{payload.question}\n\n"
        "请用中文回答，并遵循下列结构：\n"
        "1）先用 1-3 句话给出面向公众和研究人员都能理解的总体结论；\n"
        "2）然后用 2-5 条要点说明：从机制或功能角度看，这些 GO 术语（在括号中标注 GO ID 即可）如何共同支持这一结论；\n"
        "3）避免逐字重复 GO 术语的英文名称或原始英文描述，而要对其内容进行专业但通俗的转述；\n"
        "4）如果上下文不足以得出结论，请明确指出这一点，而不是勉强给出确定性表述。"
    )

    raw_answer = call_deepseek_llm(system_prompt=system_prompt, user_prompt=user_prompt)
    raw_answer = _sanitize_answer_text(raw_answer)

    sources: List[SourceItem] = []
    for score, meta in hits:
        sources.append(
            SourceItem(
                go_id=meta.get("go_id", ""),
                name=meta.get("name", ""),
                namespace=meta.get("namespace") or "",
                description=meta.get("description") or "",
                score=score,
            )
        )

    return AnswerResponse(answer=raw_answer, sources=sources)


def stream_answer_generator(payload: QuestionRequest):
    """用于 FastAPI StreamingResponse 的生成器，只流式返回答案文本。"""
    vector_store = get_vector_store(profile=payload.rag_profile)
    top_k = payload.top_k or settings.top_k

    query_for_retrieval = payload.question
    if payload.use_ner:
        try:
            query_for_retrieval = _expand_query_with_ner(
                payload.question,
                method=(payload.ner_method or "ensemble"),
                ensemble_mode=(payload.ner_ensemble_mode or "balanced"),
                ner_profile=payload.ner_profile,
            )
        except Exception:
            query_for_retrieval = payload.question

    hits = vector_store.search(query_for_retrieval, top_k=top_k)

    if not hits:
        system_prompt = (
            "你是一名面向科研人员和公众的水稻分子机理知识助手。"
            "在回答时要："
            "1）用准确、理性的中文回答；"
            "2）解释关键专业术语，避免过度专业化；"
            "3）依据不足时，要明确说明不确定性。"
        )
        user_prompt = (
            "当前没有检索到与下述问题直接关联的 GO 术语上下文，"
            "请基于你的一般生物学知识，给出客观、保守的回答。\n\n"
            f"用户问题：{payload.question}\n\n"
            "回答结构：先给出总体结论，再用 2-4 条要点解释。"
        )
    else:
        context = build_context_from_sources(hits)
        system_prompt = (
            "你是一名面向科研人员和公众的水稻分子机理知识助手。"
            "只允许基于提供的 GO 术语及其描述进行推理和回答，"
            "不要引入上下文之外的具体结论。"
            "回答要客观、理性、条理清晰，并侧重解释 GO 术语所代表的生物学含义。"
            "重要：最终回答中不要输出任何 GO ID、术语英文名/编号等标签信息，也不要逐字复述原始描述。"
        )
        user_prompt = (
            "下面是与水稻分子机理相关的一组 GO 术语及其描述，请严格以此作为事实依据，"
            "先理解这些术语对应的生物学过程或分子功能，再以面向研究人员和公众的口吻回答用户问题。\n\n"
            f"{context}\n\n"
            f"用户问题：{payload.question}\n\n"
            "请用中文回答，先给出简要结论，再用要点形式从生物学机理角度解释这些概念如何支持结论。"
            "注意：不要在回答中出现 GO ID、GO:xxxxxxx、术语英文名或编号等内容；不要逐字复述原始英文描述。"
        )

    # 逐块向前端推送内容
    tail = ""
    for chunk in stream_deepseek_llm(system_prompt=system_prompt, user_prompt=user_prompt):
        # 简单流式清洗：拼接少量尾巴以处理跨 chunk 的 "GO:xxxxxxx"
        combined = tail + chunk
        cleaned = _sanitize_answer_text(combined)
        # 保留末尾少量字符作为下次的 tail，降低跨 chunk 漏检概率
        tail = combined[-16:]
        # 输出时去掉 tail 对应的部分，避免重复
        yield cleaned[: max(0, len(cleaned) - len(tail))]

    # flush tail
    if tail:
        yield _sanitize_answer_text(tail)

