"""
src/retrieval.py — Retriever builders and strategy comparison (notebook Section 7).
"""
import logging
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger("pipeline")


def build_dense_retriever(vectorstore: Chroma, k: int = 3):
    """Standard cosine-similarity dense retriever."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    logger.info(f"Dense retriever ready (k={k})")
    return retriever


def build_mmr_retriever(
    vectorstore: Chroma,
    k: int = 3,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
):
    """
    MMR (Maximal Marginal Relevance) retriever.
    Balances relevance vs. diversity among returned chunks.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )
    logger.info(f"MMR retriever ready (k={k}, fetch_k={fetch_k}, λ={lambda_mult})")
    return retriever


def build_hybrid_retriever(
    chunks: list[Document],
    vectorstore: Chroma,
    k: int = 3,
):
    """
    Hybrid BM25 + dense EnsembleRetriever (equal weights).
    BM25 handles keyword matching; dense handles semantic similarity.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    dense = build_dense_retriever(vectorstore, k=k)

    hybrid = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5],
    )
    logger.info(f"Hybrid BM25+Dense retriever ready (k={k})")
    return hybrid


def retrieve_relevant_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 3,
    threshold: float = 0.50,
) -> list[tuple[Document, float]]:
    """
    Retrieve chunks from Chroma and filter out any whose relevance score falls below threshold.

    Args:
        vectorstore: Chroma vector store instance
        query: query string
        k: max chunks to retrieve
        threshold: minimum relevance score (0.0 to 1.0)

    Returns:
        List of (Document, score) tuples that meet or exceed the threshold.
    """
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        valid = [(doc, score) for doc, score in results if score >= threshold]
        logger.debug(
            f"Threshold retrieval: {len(valid)}/{len(results)} chunks passed threshold {threshold} for '{query[:40]}'"
        )
        return valid
    except Exception as e:
        logger.warning(f"Relevance score retrieval failed ({e}), falling back to top-{k}")
        docs = vectorstore.similarity_search(query, k=k)
        return [(d, 1.0) for d in docs]


def compare_retrieval_strategies(
    retrievers: dict,
    questions: list[dict],
    k: int = 3,
) -> pd.DataFrame:
    """
    Evaluate a dict of {name: retriever} objects and return a comparison DataFrame.

    Args:
        retrievers: {"Dense": ..., "MMR": ..., "Hybrid (BM25 + Dense)": ...}
        questions: list of evaluation question dicts with 'question' + 'expected_page'
        k: top-k cutoff

    Returns: DataFrame with columns [Strategy, Hit@3, MRR]
    """
    from src.evaluation import evaluate_retriever

    rows = []
    for name, retriever in retrievers.items():
        logger.info(f"Evaluating retrieval strategy: {name}")
        metrics = evaluate_retriever(retriever, questions, k=k)
        logger.info(f"  {name}: Hit@{k}={metrics['Hit@3']:.3f}, MRR={metrics['MRR']:.3f}")
        rows.append({"Strategy": name, **metrics})
    return pd.DataFrame(rows)


def live_compare_retrieval_strategies(
    query: str,
    vectorstore: Chroma,
    k: int = 3,
) -> dict:
    """
    Live, per-question retrieval strategy comparison that runs at **query time**
    against the already-built Chroma index.  No new embedding API calls are made —
    all three strategies do ANN / BM25 lookups against the persisted index only.

    This is intentionally separate from ``compare_retrieval_strategies()`` (the
    offline batch evaluator used by build_index.py and the indexing pipeline).

    Args:
        query:       The user's question string.
        vectorstore: An already-loaded Chroma instance (the persisted index).
        k:           Number of top chunks to retrieve per strategy.

    Returns:
        A dict keyed by strategy name with timing and retrieved chunks::

            {
              "dense":  {
                  "chunks":    [{"content": str, "paper_title": str,
                                 "page_number": str|int, "score": float}, ...],
                  "elapsed_s": float,
              },
              "mmr":    {"chunks": [...], "elapsed_s": float},
              "hybrid": {"chunks": [...], "elapsed_s": float},
            }

        Dense chunks carry a relevance score (0-1); MMR and Hybrid do not expose
        relevance scores from LangChain, so their "score" is None.
    """
    import config as _config

    # ── Build BM25 corpus once (fetched from already-persisted vectorstore) ────
    # This reuses the same pattern as _load_chain() in app.py — no API calls.
    raw = vectorstore.get()
    chunks = [
        Document(page_content=p, metadata=m)
        for p, m in zip(raw["documents"], raw["metadatas"])
    ]

    results: dict = {}

    # ── Dense (cosine similarity) ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        dense_hits = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        dense_chunks = [
            {
                "content":     doc.page_content,
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "page_number": doc.metadata.get("page_number", "?"),
                "score":       round(float(score), 4),
            }
            for doc, score in dense_hits
        ]
    except Exception as exc:
        logger.warning(f"Dense comparison failed: {exc}")
        dense_chunks = []
    results["dense"] = {"chunks": dense_chunks, "elapsed_s": time.perf_counter() - t0}

    # ── MMR ───────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        mmr_retriever = build_mmr_retriever(
            vectorstore, k=k,
            fetch_k=getattr(_config, "MMR_FETCH_K", 10),
            lambda_mult=getattr(_config, "MMR_LAMBDA", 0.5),
        )
        mmr_docs = mmr_retriever.invoke(query)
        mmr_chunks = [
            {
                "content":     doc.page_content,
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "page_number": doc.metadata.get("page_number", "?"),
                "score":       None,  # MMR doesn't expose relevance scores
            }
            for doc in mmr_docs
        ]
    except Exception as exc:
        logger.warning(f"MMR comparison failed: {exc}")
        mmr_chunks = []
    results["mmr"] = {"chunks": mmr_chunks, "elapsed_s": time.perf_counter() - t0}

    # ── Hybrid (BM25 + Dense) ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        hybrid_retriever = build_hybrid_retriever(chunks, vectorstore, k=k)
        hybrid_docs = hybrid_retriever.invoke(query)
        hybrid_chunks = [
            {
                "content":     doc.page_content,
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "page_number": doc.metadata.get("page_number", "?"),
                "score":       None,  # EnsembleRetriever doesn't expose raw scores
            }
            for doc in hybrid_docs
        ]
    except Exception as exc:
        logger.warning(f"Hybrid comparison failed: {exc}")
        hybrid_chunks = []
    results["hybrid"] = {"chunks": hybrid_chunks, "elapsed_s": time.perf_counter() - t0}

    logger.info(
        f"live_compare_retrieval_strategies: "
        f"dense={results['dense']['elapsed_s']*1000:.1f}ms "
        f"mmr={results['mmr']['elapsed_s']*1000:.1f}ms "
        f"hybrid={results['hybrid']['elapsed_s']*1000:.1f}ms"
    )
    return results

