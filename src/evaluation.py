"""
src/evaluation.py — Evaluation questions and metric functions (notebook Section 3 & 8).

Contains:
- evaluation_questions  : 23 tuning questions with expected_page ground truth
- test_questions        : 10 held-out test strings
- Metric functions      : hit_at_k, reciprocal_rank
- Strategy evaluators   : evaluate_chunking_strategy, evaluate_retriever
- Helper                : load_generated_questions (merges LLM-generated Qs)
"""
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

logger = logging.getLogger("pipeline")

# ── Tuning set (23 questions from notebook cell 18) ───────────────────────────
evaluation_questions: list[dict] = [
    {"question": "What is Agentic Retrieval-Augmented Generation (Agentic RAG) and how does it differ from traditional RAG?", "expected_page": 1},
    {"question": "What are the three primary components of a RAG system's architecture?", "expected_page": 3},
    {"question": "What are the limitations of Naïve RAG?", "expected_page": 4},
    {"question": "What key innovations does Modular RAG introduce?", "expected_page": 5},
    {"question": "What are the key characteristics and challenges of Agentic RAG as described in the evolution of RAG paradigms?", "expected_page": 7},
    {"question": "What are the four components that make up an AI agent?", "expected_page": 9},
    {"question": "What is the Reflection design pattern in agentic workflows?", "expected_page": 9},
    {"question": "What is the difference between Prompt Chaining and Routing workflow patterns?", "expected_page": 11},
    {"question": "What is the Orchestrator-Workers workflow pattern and when should it be used?", "expected_page": 13},
    {"question": "What is the workflow of a Single-Agent Agentic RAG (Router) system?", "expected_page": 14},
    {"question": "What are the key features and challenges of Multi-Agent Agentic RAG systems?", "expected_page": 17},
    {"question": "How does Hierarchical Agentic RAG organize its agents and what is its workflow?", "expected_page": 18},
    {"question": "What are the five key agents in the Corrective RAG system?", "expected_page": 20},
    {"question": "How does Adaptive RAG dynamically adjust its query handling strategy?", "expected_page": 21},
    {"question": "What is Agent-G and how does it integrate graph knowledge bases with unstructured document retrieval?", "expected_page": 23},
    {"question": "What are the two primary innovations of GeAR (Graph-Enhanced Agent for Retrieval-Augmented Generation)?", "expected_page": 25},
    {"question": "What is the workflow of Agentic Document Workflows (ADW)?", "expected_page": 27},
    {"question": "How do Traditional RAG, Agentic RAG, and Agentic Document Workflows compare in terms of context maintenance and scalability?", "expected_page": 29},
    {"question": "How is Agentic RAG applied in healthcare and personalized medicine?", "expected_page": 30},
    {"question": "What tools and frameworks support the development of Agentic RAG systems?", "expected_page": 31},
    {"question": "What practical lessons are given regarding when Agentic RAG should or should not be used?", "expected_page": 32},
    {"question": "What open research challenges exist around agent coordination and evaluation methodologies in Agentic RAG?", "expected_page": 34},
    {"question": "What benchmarks are commonly used to evaluate RAG systems, such as BEIR and HotpotQA?", "expected_page": 36},
]

# ── Held-out test set (10 questions from notebook cell 18) ────────────────────
test_questions: list[str] = [
    "What is Agentic Retrieval-Augmented Generation?",
    "Explain GraphRAG and its two main innovations.",
    "What evaluation benchmarks are used for RAG systems?",
    "What is the difference between Naive RAG and Modular RAG?",
    "Describe the Reflection design pattern.",
    "What is the Orchestrator-Workers pattern used for?",
    "How does Corrective RAG decide when to fall back to web search?",
    "What are the main challenges of Multi-Agent Agentic RAG?",
    "How is Agentic RAG used in healthcare?",
    "What open research challenges remain in agent evaluation?",
]


# ── Metric primitives ─────────────────────────────────────────────────────────

def hit_at_k(results: list[dict], expected_page: int, k: int = 3) -> bool:
    """True if expected_page appears in the top-k retrieved chunks."""
    pages = [r["chunk"].metadata.get("page_number") for r in results[:k]]
    return expected_page in pages


def reciprocal_rank(results: list[dict], expected_page: int) -> float:
    """1/rank of the first hit, or 0 if not found."""
    for rank, r in enumerate(results, start=1):
        if r["chunk"].metadata.get("page_number") == expected_page:
            return 1 / rank
    return 0.0


# ── Chunking evaluation (raw cosine — for the 3×2 grid) ──────────────────────

def _retrieve_top_k_cosine(
    query: str,
    chunks: list[Document],
    embeddings: list,
    embedding_model,
    k: int = 3,
) -> list[dict]:
    query_vec = embedding_model.embed_query(query)
    sims = cosine_similarity([query_vec], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [{"chunk": chunks[idx], "score": float(sims[idx])} for idx in top_idx]


def evaluate_chunking_strategy(
    chunks: list[Document],
    embeddings: list,
    questions: list[dict],
    embedding_model,
    k: int = 3,
) -> dict:
    """
    Evaluate a chunking strategy using raw cosine similarity (no vector DB).
    Used for the 3-strategy × 2-embedding grid comparison.

    Returns: {"Hit@3": float, "MRR": float}
    """
    hits, mrrs = [], []
    for item in questions:
        results = _retrieve_top_k_cosine(
            item["question"], chunks, embeddings, embedding_model, k=k
        )
        hits.append(hit_at_k(results, item["expected_page"], k=k))
        mrrs.append(reciprocal_rank(results, item["expected_page"]))
    return {"Hit@3": round(sum(hits) / len(hits), 4),
            "MRR":   round(sum(mrrs) / len(mrrs), 4)}


# ── Retrieval evaluation (works with any LangChain retriever) ─────────────────

def evaluate_retriever(retriever, questions: list[dict], k: int = 3) -> dict:
    """
    Evaluate any LangChain retriever object using Hit@k / MRR.

    Args:
        retriever: any object with an .invoke(query) method.
        questions: list of dicts with 'question' and 'expected_page'.
        k: number of top results to consider.

    Returns: {"Hit@3": float, "MRR": float}
    """
    hits, mrrs = [], []
    for item in questions:
        docs = retriever.invoke(item["question"])[:k]
        pages = [d.metadata.get("page_number") for d in docs]
        hits.append(item["expected_page"] in pages)
        rr = 0.0
        for rank, page in enumerate(pages, start=1):
            if page == item["expected_page"]:
                rr = 1 / rank
                break
        mrrs.append(rr)
    return {"Hit@3": round(sum(hits) / len(hits), 4),
            "MRR":   round(sum(mrrs) / len(mrrs), 4)}


# ── Generated-question loader ─────────────────────────────────────────────────

def load_generated_questions(path) -> list[dict]:
    """
    Load LLM-generated questions from config/generated_questions.json.
    Returns [] if the file does not exist or is malformed.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = [
            q for q in data
            if isinstance(q, dict) and "question" in q and "expected_page" in q
        ]
        logger.info(f"Loaded {len(valid)} generated questions from {path.name}")
        return valid
    except Exception as e:
        logger.warning(f"Could not load generated questions: {e}")
        return []
