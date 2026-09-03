"""
src/rag_chain.py — Single-turn RAG chain with consistent source citations (notebook Section 6).

ask() is used identically by scripts/evaluate.py and app.py so citation
behaviour is always consistent, not duplicated.
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
    llm = None,
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

    Returns:
        {
            "answer": str,
            "sources": [{"paper_title": str, "page_number": int}, ...],
            "web_sources": [{"title": str, "url": str, "content": str}, ...]
        }
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
    return {"answer": answer, "sources": sources, "web_sources": web_sources}
