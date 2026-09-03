"""
scripts/evaluate.py — Standalone evaluation deliverable.

Loads the persisted config + vector store, merges static test_questions
with LLM-generated questions, runs every question through ask(), and
reports aggregate Hit@3 / MRR.

Run:
    python scripts/evaluate.py

This is the "Testing & Evaluation" rubric deliverable — demoable in a
terminal during a mentor review, no Streamlit needed.
"""
import json
import os
import sys
from pathlib import Path

# ── Make src/ importable when called as python scripts/evaluate.py ────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import config
from src.run_logger import setup_console_logging
from src.embeddings import get_embedding_model
from src.vectorstore import load_vectorstore
from src.retrieval import build_dense_retriever, build_mmr_retriever, build_hybrid_retriever
from src.rag_chain import build_rag_chain, ask
from src.evaluation import test_questions, load_generated_questions, evaluate_retriever

setup_console_logging()

import logging
logger = logging.getLogger("pipeline")


def main():
    # ── Load config ──────────────────────────────────────────────────────────
    if not config.BEST_CONFIG_PATH.exists():
        print("ERROR: config/best_config.json not found.")
        print("Run  python scripts/build_index.py  first.")
        sys.exit(1)

    with open(config.BEST_CONFIG_PATH, "r") as f:
        best_cfg = json.load(f)

    emb_key         = best_cfg["embedding_model"]
    chunking        = best_cfg["chunking_strategy"]
    retrieval_name  = best_cfg["retrieval_strategy"]

    print(f"\n{'═'*70}")
    print("  EVALUATION — Research Paper Answer Bot")
    print(f"{'═'*70}")
    print(f"  Chunking:  {chunking}")
    print(f"  Embedding: {emb_key}")
    print(f"  Retrieval: {retrieval_name}")
    print(f"{'═'*70}\n")

    # ── Load model & vector store ────────────────────────────────────────────
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    emb_model   = get_embedding_model(emb_key)
    vectorstore = load_vectorstore(emb_model, config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME)
    llm         = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMP)

    # ── Build retriever ──────────────────────────────────────────────────────
    if retrieval_name == "MMR":
        retriever = build_mmr_retriever(
            vectorstore, k=config.TOP_K,
            fetch_k=config.MMR_FETCH_K, lambda_mult=config.MMR_LAMBDA,
        )
    elif "Hybrid" in retrieval_name:
        # Hybrid needs access to all chunks — load from the collection
        all_docs = vectorstore.get()
        from langchain_core.documents import Document
        chunks = [
            Document(page_content=pc, metadata=meta)
            for pc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        retriever = build_hybrid_retriever(chunks, vectorstore, k=config.TOP_K)
    else:
        retriever = build_dense_retriever(vectorstore, k=config.TOP_K)

    chain = build_rag_chain(retriever, llm)

    # ── Build question set ───────────────────────────────────────────────────
    static_qs = [{"question": q, "expected_page": None} for q in test_questions]
    generated_qs = load_generated_questions(config.GENERATED_QUESTIONS_PATH)
    all_questions = static_qs + generated_qs

    print(f"Running {len(all_questions)} questions "
          f"({len(static_qs)} static + {len(generated_qs)} generated)…\n")

    # ── Run answers ──────────────────────────────────────────────────────────
    hits, mrrs = [], []
    for i, item in enumerate(all_questions, start=1):
        q = item["question"]
        expected = item.get("expected_page")

        result = ask(chain, retriever, q, k=config.TOP_K)
        answer  = result["answer"]
        sources = result["sources"]

        print(f"{'─'*70}")
        print(f"Q{i:02d}: {q}")
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for s in sources:
            print(f"  • {s['paper_title']} — page {s['page_number']}")

        if expected is not None:
            retrieved_pages = [s["page_number"] for s in sources]
            hit = expected in retrieved_pages
            rr  = next((1/(r+1) for r, p in enumerate(retrieved_pages) if p == expected), 0.0)
            hits.append(hit)
            mrrs.append(rr)
            print(f"\n  [Eval] expected_page={expected}  hit={hit}  RR={rr:.3f}")
        print()

    # ── Aggregate metrics ────────────────────────────────────────────────────
    if hits:
        print(f"{'═'*70}")
        print(f"  AGGREGATE  (over {len(hits)} questions with ground truth)")
        print(f"  Hit@{config.TOP_K}: {sum(hits)/len(hits):.3f}")
        print(f"  MRR:   {sum(mrrs)/len(mrrs):.3f}")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
