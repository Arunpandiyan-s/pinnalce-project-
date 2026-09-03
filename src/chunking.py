"""
src/chunking.py — Three text chunking strategies (notebook Section 2).
Each chunk is tagged with chunk_id and chunking_strategy metadata.
"""
import logging
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger("pipeline")


def fixed_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents with a fixed-size character splitter."""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"fixed_chunk_{i}"
        chunk.metadata["chunking_strategy"] = "fixed"
    logger.info(f"Fixed chunks: {len(chunks)}")
    return chunks


def recursive_chunk(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents with a recursive character splitter (respects paragraph/sentence boundaries)."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"recursive_chunk_{i}"
        chunk.metadata["chunking_strategy"] = "recursive"
    logger.info(f"Recursive chunks: {len(chunks)}")
    return chunks


def semantic_chunk(
    documents: list[Document],
    embedding_model,
) -> list[Document]:
    """Split documents using SemanticChunker (groups semantically similar sentences)."""
    from langchain_experimental.text_splitter import SemanticChunker  # lazy import

    splitter = SemanticChunker(embedding_model)
    chunks = splitter.split_documents(documents)

    page_counter: dict = defaultdict(int)
    for chunk in chunks:
        page = chunk.metadata.get("page_number", 0)
        page_counter[page] += 1
        doc_id = chunk.metadata.get("document_id", "doc")
        chunk.metadata["chunk_id"] = (
            f"{doc_id}_page_{page:03d}_chunk_{page_counter[page]:03d}"
        )
        chunk.metadata["chunking_strategy"] = "semantic"
    logger.info(f"Semantic chunks: {len(chunks)}")
    return chunks


def get_all_chunking_strategies(
    documents: list[Document],
    embedding_model,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict[str, list[Document]]:
    """
    Apply all three chunking strategies and return a dict keyed by strategy name.

    Returns:
        {"fixed": [...], "recursive": [...], "semantic": [...]}
    """
    logger.info("Generating all chunking strategies…")
    strategies = {
        "fixed":     fixed_chunk(documents, chunk_size, chunk_overlap),
        "recursive": recursive_chunk(documents, chunk_size, chunk_overlap),
        "semantic":  semantic_chunk(documents, embedding_model),
    }
    return strategies
