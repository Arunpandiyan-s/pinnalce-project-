"""
src/retrieval.py — Retriever builders and strategy comparison (notebook Section 7).
"""
import logging
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

