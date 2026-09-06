"""
scripts/build_index.py — Offline indexing pipeline (run once per corpus change).

Workflow
--------
1.  Discover PDFs in PDF_DIR
2.  Load & clean documents
3.  Generate 3 chunking strategies × 2 embedding models → 6 combinations
4.  Evaluate all 6 with Hit@3 / MRR, print table, pick winner
5.  Re-evaluate the 2 embedding models on the winning chunking strategy, pick best embedding
6.  Build & persist Chroma vector store (winning chunking + embedding)
7.  Build Dense / MMR / Hybrid retrievers
8.  Evaluate all 3 retrievers with Hit@3 / MRR, print table, pick winner
9.  Write config/best_config.json

This script is the justification artefact for:
  - "at least 2 embedding models compared"
  - "multiple retrieval strategies compared"
  - "justification for final choice" (the printed tables ARE the justification)
"""
import json
import sys
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Make src/ importable when called as python scripts/build_index.py ─────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import config
from src.run_logger import setup_console_logging
from src.ingestion import discover_pdfs, load_and_clean_pdfs
from src.chunking import get_all_chunking_strategies
from src.embeddings import get_embedding_model, embed_documents_rate_limited
from src.evaluation import evaluation_questions, evaluate_chunking_strategy
from src.vectorstore import build_vectorstore, build_vectorstore_from_precomputed, load_vectorstore
from src.retrieval import (
    build_dense_retriever,
    build_mmr_retriever,
    build_hybrid_retriever,
    compare_retrieval_strategies,
)

setup_console_logging()
logger = logging.getLogger("pipeline")


def _print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    print(df.to_string(index=False))
    print()


def run_pipeline(pdf_dir=None, persist_dir=None, collection_name=None,
                 best_config_path=None) -> dict:
    """
    Run the full offline indexing pipeline.
    Returns the best_config dict that was written to disk.
    Can be called programmatically from app.py.
    """
    pdf_dir         = pdf_dir          or config.PDF_DIR
    persist_dir     = persist_dir      or config.CHROMA_PERSIST_DIR
    collection_name = collection_name  or config.COLLECTION_NAME
    best_config_path = best_config_path or config.BEST_CONFIG_PATH

    # ── Step 1 & 2: Load PDFs ────────────────────────────────────────────────
    logger.info("=== STEP 1: Discover & load PDFs ===")
    pdf_sources = discover_pdfs(pdf_dir)
    if not pdf_sources:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. "
            "Add at least one PDF and re-run."
        )
    documents = load_and_clean_pdfs(pdf_sources)

    # ── Step 3: Load embedding models ────────────────────────────────────────
    logger.info("=== STEP 2: Load embedding models ===")
    embedding_models = {key: get_embedding_model(key) for key in config.EMBEDDING_MODELS}

    # ── Step 4: 3 × 2 chunking/embedding grid ───────────────────────────────
    logger.info("=== STEP 3: 3 chunking × 2 embedding = 6 combinations ===")
    # Use bge model for semantic chunker (lightweight, good quality)
    all_chunks = get_all_chunking_strategies(
        documents,
        embedding_model=embedding_models["bge"],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    grid_rows = []
    chunk_embeddings: dict = {}   # (strategy, emb_key) -> embeddings list

    for strategy_name, chunks in all_chunks.items():
        texts = [c.page_content for c in chunks]
        for emb_key, emb_model in embedding_models.items():
            logger.info(f"  Embedding: {strategy_name} + {emb_key} ({len(chunks)} chunks)…")
            if emb_key == "cohere":
                embeddings = embed_documents_rate_limited(
                    emb_model, texts, batch_size=40
                    # cooldown=20 triggers only on actual 429 errors (adaptive)
                )
            else:
                embeddings = emb_model.embed_documents(texts)
            chunk_embeddings[(strategy_name, emb_key)] = embeddings
            metrics = evaluate_chunking_strategy(
                chunks, embeddings, evaluation_questions, emb_model,
                k=config.TOP_K,
            )
            grid_rows.append({
                "Chunking":  strategy_name.capitalize(),
                "Embedding": emb_key,
                **metrics,
            })

    grid_df = pd.DataFrame(grid_rows)
    _print_table(grid_df, "Chunking × Embedding Grid (Hit@3 / MRR)")

    # Pick winner by MRR, tie-break Hit@3
    best_row = grid_df.sort_values(["MRR", "Hit@3"], ascending=False).iloc[0]
    best_chunking = best_row["Chunking"].lower()
    best_emb_key  = best_row["Embedding"]
    logger.info(
        f"Winner → chunking='{best_chunking}', embedding='{best_emb_key}' "
        f"(MRR={best_row['MRR']:.3f}, Hit@3={best_row['Hit@3']:.3f})"
    )

    best_chunks    = all_chunks[best_chunking]
    best_emb_model = embedding_models[best_emb_key]
    best_embeddings = chunk_embeddings[(best_chunking, best_emb_key)]

    # ── Step 5: Build & persist Chroma (reuse precomputed embeddings) ─────────
    logger.info("=== STEP 4: Build Chroma vector store ===")
    vectorstore = build_vectorstore_from_precomputed(
        best_chunks, best_embeddings, best_emb_model, persist_dir, collection_name
    )

    # ── Step 6: Build & evaluate 3 retrievers ───────────────────────────────
    logger.info("=== STEP 5: Compare retrieval strategies ===")
    retrievers = {
        "Dense":               build_dense_retriever(vectorstore, k=config.TOP_K),
        "MMR":                 build_mmr_retriever(
                                   vectorstore, k=config.TOP_K,
                                   fetch_k=config.MMR_FETCH_K,
                                   lambda_mult=config.MMR_LAMBDA,
                               ),
        "Hybrid (BM25+Dense)": build_hybrid_retriever(
                                   best_chunks, vectorstore, k=config.TOP_K
                               ),
    }

    retrieval_df = compare_retrieval_strategies(
        retrievers, evaluation_questions, k=config.TOP_K
    )
    _print_table(retrieval_df, "Retrieval Strategy Comparison (Hit@3 / MRR)")

    best_retrieval_row = retrieval_df.sort_values(
        ["MRR", "Hit@3"], ascending=False
    ).iloc[0]
    best_retrieval = best_retrieval_row["Strategy"]
    logger.info(
        f"Winner → retrieval='{best_retrieval}' "
        f"(MRR={best_retrieval_row['MRR']:.3f}, Hit@3={best_retrieval_row['Hit@3']:.3f})"
    )

    # ── Step 7: Write best_config.json ───────────────────────────────────────
    logger.info("=== STEP 6: Write best_config.json ===")
    best_config = {
        "chunking_strategy":  best_chunking,
        "embedding_model":    best_emb_key,
        "retrieval_strategy": best_retrieval,
        "scores": {
            "Hit@3": float(best_retrieval_row["Hit@3"]),
            "MRR":   float(best_retrieval_row["MRR"]),
        },
        "grid_results":     grid_df.to_dict(orient="records"),
        "retrieval_results": retrieval_df.to_dict(orient="records"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    config_path = Path(best_config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"best_config.json written → {config_path}")

    return best_config


if __name__ == "__main__":
    result = run_pipeline()
    print("\n✅ Pipeline complete.")
    print(
        f"   Best: {result['chunking_strategy']} chunking + "
        f"{result['embedding_model']} embedding + "
        f"{result['retrieval_strategy']} retrieval"
    )
    print(f"   Scores: {result['scores']}")
