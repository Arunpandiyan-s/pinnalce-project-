"""
src/rag_chain.py — Single-turn RAG chain with consistent source citations (notebook Section 6).

ask() is used identically by scripts/evaluate.py and app.py so citation
behaviour is always consistent, not duplicated.

Compare mode
------------
Pass ``compare_mode=True`` to ``ask()`` to get a side-by-side comparison of
Dense / MMR / Hybrid retrieval results for the same query, returned under the
``"comparison"`` key.  The normal answer (default strategy) is still generated
and returned at the top level.  No new embedding API calls are made — all three
strategies run against the already-persisted Chroma index.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logger = logging.getLogger("pipeline")

_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a research assistant answering questions about academic papers.

Answer the question using ONLY the provided context below.
If the context does not contain enough information to answer confidently,
say "I don't know based on the provided context" instead of guessing.

Context:
{context}

Question:
{question}

Answer:"""
)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('paper_title')}, p.{d.metadata.get('page_number')}]\n"
        f"{d.page_content}"
        for d in docs
    )


def build_rag_chain(retriever, llm):
    """
    Build a grounded single-turn RAG chain using LCEL.

    Returns:
        A Runnable chain that accepts a question string and returns an answer string.
    """
    chain = (
        {
            "context": retriever | _format_context,
            "question": lambda x: x,
        }
        | _PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    logger.debug("Single-turn RAG chain built")
    return chain


def ask(
    chain,
    retriever,
    question: str,
    k: int = 3,
    use_web_search: bool = False,
    llm=None,
    compare_mode: bool = False,
    generate_per_strategy_answers: bool = False,
    vectorstore=None,
) -> dict:
    """
    Run the RAG chain and return a structured response with answer, paper sources, and web sources.

    Args:
        chain: built by build_rag_chain()
        retriever: the retriever used to fetch supporting docs
        question: user question string
        k: number of source documents to include
        use_web_search: whether to search Tavily for web sources
        llm: optional LLM model for synthesis when web search is enabled
        compare_mode: when True, runs all three retrieval strategies against the
            already-built vectorstore and returns per-strategy chunks + timing
            under the "comparison" key.  No new embedding API calls are made.
            The normal final answer (from the configured default retriever) is
            still generated and returned at the top level.
        generate_per_strategy_answers: when True (and compare_mode=True), also
            generates one short LLM answer per strategy using that strategy's
            retrieved chunks as context.  This triggers 3× LLM calls instead of
            1 — opt-in only.  Ignored when compare_mode=False.
        vectorstore: the Chroma vectorstore instance, required when
            compare_mode=True so that live_compare_retrieval_strategies() can
            build BM25 corpus and do ANN lookups without any API calls.

    Returns:
        When compare_mode=False (default)::

            {
                "answer": str,
                "sources": [{"paper_title": str, "page_number": int, "chunk_id": str}, ...],
                "web_sources": [{"title": str, "url": str, "content": str}, ...],
            }

        When compare_mode=True, the dict also contains::

            "comparison": {
                "dense":  {"chunks": [...], "elapsed_s": float, "answer": str | None},
                "mmr":    {"chunks": [...], "elapsed_s": float, "answer": str | None},
                "hybrid": {"chunks": [...], "elapsed_s": float, "answer": str | None},
            }

        Each chunk dict: {"content": str, "paper_title": str, "page_number": str|int,
        "score": float | None}
    """
    docs = retriever.invoke(question)[:k] if retriever else []
    web_sources = []

    if use_web_search:
        from src.web_search import tavily_search, format_web_context, is_tavily_available
        if is_tavily_available():
            web_results = tavily_search(question)
            web_sources = web_results
            if web_results and llm:
                paper_ctx = _format_context(docs)
                web_ctx = format_web_context(web_results)
                combined_ctx = f"--- Academic Papers Context ---\n{paper_ctx}\n\n--- Live Web Context ---\n{web_ctx}"
                prompt = (
                    f"You are a research assistant answering questions using both academic papers and live web search.\n\n"
                    f"Context:\n{combined_ctx}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer accurately and synthesize information from both paper and web contexts where applicable:"
                )
                answer = llm.invoke(prompt).content
            else:
                answer = chain.invoke(question)
        else:
            logger.warning("Web search requested but Tavily is not available.")
            answer = chain.invoke(question)
    else:
        answer = chain.invoke(question)

    sources = [
        {
            "paper_title": d.metadata.get("paper_title", "Unknown"),
            "page_number": d.metadata.get("page_number", "?"),
            "chunk_id":    d.metadata.get("chunk_id", ""),
        }
        for d in docs
    ]

    logger.debug(
        f"Q: {question[:60]}… → {len(sources)} paper sources, {len(web_sources)} web sources cited"
    )

    result = {"answer": answer, "sources": sources, "web_sources": web_sources}

    # ── Compare mode: live per-strategy retrieval (no embedding calls) ─────────
    if compare_mode and vectorstore is not None:
        from src.retrieval import live_compare_retrieval_strategies
        logger.info("compare_mode=True — running live retrieval strategy comparison")

        comparison = live_compare_retrieval_strategies(question, vectorstore, k=k)

        if generate_per_strategy_answers and llm is not None:
            # Opt-in: generate one short answer per strategy (3× LLM calls)
            _per_strategy_prompt = ChatPromptTemplate.from_template(
                "You are a research assistant. Using ONLY the context below, answer the question "
                "in 2-3 sentences.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            for strategy_name, strategy_data in comparison.items():
                chunks = strategy_data["chunks"]
                if chunks:
                    ctx_text = "\n\n".join(
                        f"[{c['paper_title']}, p.{c['page_number']}]\n{c['content']}"
                        for c in chunks
                    )
                    try:
                        per_ans = (
                            _per_strategy_prompt
                            | llm
                            | StrOutputParser()
                        ).invoke({"context": ctx_text, "question": question})
                    except Exception as exc:
                        logger.warning(f"Per-strategy answer ({strategy_name}) failed: {exc}")
                        per_ans = None
                else:
                    per_ans = None
                comparison[strategy_name]["answer"] = per_ans
        else:
            # No per-strategy answers — set answer=None on each strategy
            for strategy_data in comparison.values():
                strategy_data["answer"] = None

        result["comparison"] = comparison
        logger.info("compare_mode: comparison data added to result")

    return result
