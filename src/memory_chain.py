"""
src/memory_chain.py — Multi-turn conversational RAG chain (notebook Section 9, Stretch Goal 1).

Uses create_history_aware_retriever to rewrite follow-up questions
into standalone queries before retrieval, so pronoun references work.
"""
import logging
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("pipeline")

_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, rewrite the "
        "question as a self-contained standalone question. "
        "Do NOT answer the question — only rewrite it.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'I don't know based on the provided context'.\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_conversational_chain(retriever, llm):
    """
    Build a multi-turn conversational RAG chain.

    Args:
        retriever: any LangChain retriever
        llm: a ChatGoogleGenerativeAI (or any ChatModel)

    Returns:
        A Runnable that accepts {"input": str, "chat_history": list} and
        returns {"answer": str, "context": list[Document], ...}
    """
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, _CONTEXTUALIZE_PROMPT
    )
    document_chain = create_stuff_documents_chain(llm, _QA_PROMPT)
    chain = create_retrieval_chain(history_aware_retriever, document_chain)
    logger.debug("Conversational RAG chain built")
    return chain


def chat(
    chain,
    chat_history: list,
    question: str,
    use_web_search: bool = False,
    llm = None,
    vectorstore = None,
    threshold: float = 0.50,
    selected_papers: list | None = None,
) -> dict:
    """
    Run one turn of the conversational chain with adaptive 3-tier retrieval,
    similarity thresholding, and source attribution.

    Args:
        chain: built by build_conversational_chain()
        chat_history: list of HumanMessage / AIMessage objects (LangChain format)
        question: current user question
        use_web_search: whether to fetch live web search results from Tavily
        llm: ChatModel instance for answer synthesis
        vectorstore: optional Chroma vectorstore to evaluate relevance scores
        threshold: minimum relevance score (default 0.50)
        selected_papers: optional list of paper_title strings to restrict retrieval to.
                         If None or empty, all documents are searched.

    Returns:
        {
            "answer": str,
            "sources": [{"paper_title": str, "page_number": int}, ...],
            "web_sources": [{"title": str, "url": str, "content": str}, ...],
            "source_type": "papers" | "web" | "hybrid" | "llm",
            "chat_history": updated list (with this turn appended)
        }
    """
    import os
    import config
    threshold = getattr(config, "SIMILARITY_THRESHOLD", threshold)

    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMP, google_api_key=api_key)

    # ── 1. Build optional document filter ─────────────────────────────────────
    # Chroma where-filter restricts search to selected paper titles only.
    where_filter = None
    if selected_papers:
        if len(selected_papers) == 1:
            where_filter = {"paper_title": selected_papers[0]}
        else:
            where_filter = {"paper_title": {"$in": selected_papers}}
        logger.info(f"Document filter active: {selected_papers}")

    # ── 2. Evaluate vector search relevance ──────────────────────────────────
    valid_paper_docs = []
    if vectorstore is not None:
        try:
            results = vectorstore.similarity_search_with_relevance_scores(
                question,
                k=config.TOP_K,
                filter=where_filter if where_filter else None,
            )
            valid_paper_docs = [doc for doc, score in results if score >= threshold]
            logger.info(
                f"Vector relevance check: {len(valid_paper_docs)}/{len(results)} chunks above threshold {threshold}"
            )
        except Exception as e:
            logger.warning(f"Vector threshold search failed ({e}), using standard retriever")
            result = chain.invoke({"input": question, "chat_history": chat_history})
            valid_paper_docs = result.get("context", [])[:config.TOP_K]
    else:
        # Fallback if vectorstore not passed directly
        result = chain.invoke({"input": question, "chat_history": chat_history})
        valid_paper_docs = result.get("context", [])[:config.TOP_K]

    has_paper_matches = len(valid_paper_docs) > 0

    # ── 2. Determine Web Search results ──────────────────────────────────────
    web_results = []
    if use_web_search:
        from src.web_search import tavily_search, is_tavily_available
        if is_tavily_available():
            web_results = tavily_search(question)
        else:
            logger.warning("Web search requested but Tavily is not configured.")

    has_web_matches = len(web_results) > 0

    # ── 3. Determine Source Attribution Tier & Generate Answer ───────────────
    paper_sources = [
        {
            "paper_title": d.metadata.get("paper_title", "Unknown"),
            "page_number": d.metadata.get("page_number", "?"),
        }
        for d in valid_paper_docs
    ]

    from src.web_search import format_web_context

    if has_paper_matches and has_web_matches:
        # Tier: Hybrid (Both Paper + Live Web)
        source_type = "hybrid"
        paper_ctx = "\n\n".join(
            f"[{d.metadata.get('paper_title', 'Unknown')}, p.{d.metadata.get('page_number', '?')}]\n{d.page_content}"
            for d in valid_paper_docs
        )
        web_ctx = format_web_context(web_results)
        combined_ctx = f"--- Academic Papers Context ---\n{paper_ctx}\n\n--- Live Web Context ---\n{web_ctx}"

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a research assistant answering questions using both academic papers and live web search.\n"
                "Synthesize information accurately from both contexts where relevant.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        gen_chain = prompt_template | llm
        res = gen_chain.invoke({
            "context": combined_ctx,
            "chat_history": chat_history,
            "input": question,
        })
        answer = res.content
        sources = paper_sources
        web_sources = web_results

    elif has_paper_matches and not has_web_matches:
        # Tier 1: Pure Paper RAG
        source_type = "papers"
        paper_ctx = "\n\n".join(
            f"[{d.metadata.get('paper_title', 'Unknown')}, p.{d.metadata.get('page_number', '?')}]\n{d.page_content}"
            for d in valid_paper_docs
        )
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a research assistant answering questions about academic papers.\n"
                "Answer ONLY using the provided context below.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        gen_chain = prompt_template | llm
        res = gen_chain.invoke({
            "context": paper_ctx,
            "chat_history": chat_history,
            "input": question,
        })
        answer = res.content
        sources = paper_sources
        web_sources = []

    elif not has_paper_matches and has_web_matches:
        # Tier 2: Live Web Search Only (No false paper citations!)
        source_type = "web"
        web_ctx = format_web_context(web_results)
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an assistant answering questions using live web search results.\n"
                "Answer the user's question accurately using the web context below.\n\n"
                "Web Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        gen_chain = prompt_template | llm
        res = gen_chain.invoke({
            "context": web_ctx,
            "chat_history": chat_history,
            "input": question,
        })
        answer = res.content
        sources = []
        web_sources = web_results

    else:
        # Tier 3: General LLM Knowledge
        source_type = "llm"
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant. The user's question was not found in the uploaded research papers.\n"
                "Answer the question clearly using your general knowledge.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        gen_chain = prompt_template | llm
        res = gen_chain.invoke({
            "chat_history": chat_history,
            "input": question,
        })
        answer = res.content
        sources = []
        web_sources = []

    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]

    logger.debug(
        f"Chat turn processed [{source_type}] — {len(sources)} paper sources, {len(web_sources)} web sources cited"
    )
    return {
        "answer": answer,
        "sources": sources,
        "web_sources": web_sources,
        "source_type": source_type,
        "chat_history": updated_history,
    }

