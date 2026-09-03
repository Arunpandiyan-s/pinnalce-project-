"""
src/vectorstore.py — Build and load persisted Chroma vector stores.
"""
import logging
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger("pipeline")


def build_vectorstore(
    chunks: list[Document],
    embedding_model,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Build a Chroma vector store from chunks and persist it to disk.
    Deletes any existing store at persist_dir first to avoid stale data.

    Returns: loaded Chroma instance.
    """
    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        logger.info(f"Cleared existing vector store at {persist_dir}")

    # Strip raw embedding vectors from metadata before indexing
    for chunk in chunks:
        chunk.metadata.pop("embedding", None)

    logger.info(f"Indexing {len(chunks)} chunks into Chroma…")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )
    vectorstore.add_documents(chunks)
    count = vectorstore._collection.count()
    logger.info(f"Vector store ready — {count} embeddings persisted to {persist_dir}")
    return vectorstore


def build_vectorstore_from_precomputed(
    chunks: list[Document],
    embeddings: list[list[float]],
    embedding_model,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Persist a Chroma vector store using *already-computed* embeddings, avoiding
    a redundant embedding pass.

    Uses chromadb's Collection.add() directly (chromadb >=0.5 / 1.x API):
        collection.add(ids, embeddings, metadatas, documents)
    The Chroma wrapper is still constructed with ``embedding_function`` so that
    query-time similarity search (which calls embed_query on the fly) continues
    to work correctly.

    Args:
        chunks:          LangChain Document objects (must be same length and order
                         as *embeddings*).
        embeddings:      Pre-computed vectors — one per chunk, same order.
        embedding_model: Used for query-time embedding only (not re-indexed here).
        persist_dir:     Directory to persist the Chroma DB.
        collection_name: Chroma collection name.

    Returns:
        Chroma instance backed by the newly persisted collection.

    Raises:
        ValueError: if len(chunks) != len(embeddings).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have "
            "the same length."
        )

    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        logger.info(f"Cleared existing vector store at {persist_dir}")

    # Strip raw embedding vectors that may have been stored in metadata
    for chunk in chunks:
        chunk.metadata.pop("embedding", None)

    logger.info(
        f"Building vector store from {len(chunks)} precomputed embeddings…"
    )

    # Create the LangChain Chroma wrapper (empty — no add_documents yet)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

    # Insert directly into the underlying chromadb collection.
    # chromadb 1.x Collection.add() signature:
    #   add(ids, embeddings, metadatas, documents) — all positional or keyword.
    # Metadatas must not contain None values; filter them out per field.
    ids = [str(i) for i in range(len(chunks))]
    metadatas: list[dict[str, str | int | float | bool]] = [
        {k: v for k, v in (chunk.metadata or {}).items() if v is not None}
        for chunk in chunks
    ]
    documents = [chunk.page_content for chunk in chunks]

    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings,  # type: ignore[arg-type]  # list[list[float]] satisfies PyEmbedding at runtime
        metadatas=metadatas,  # type: ignore[arg-type]  # dict[str, str|int|float|bool] satisfies Metadata at runtime
        documents=documents,
    )

    count = vectorstore._collection.count()
    logger.info(
        f"Vector store ready — {count} embeddings persisted to {persist_dir} "
        "(no re-embedding performed)"
    )
    return vectorstore


def load_vectorstore(
    embedding_model,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """
    Load an existing persisted Chroma vector store (no re-indexing).
    Used by app.py and scripts/evaluate.py at serve time.

    Returns: Chroma instance backed by the persisted collection.
    """
    logger.info(f"Loading vector store from {persist_dir}")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )
    count = vectorstore._collection.count()
    logger.info(f"Vector store loaded — {count} embeddings")
    return vectorstore
