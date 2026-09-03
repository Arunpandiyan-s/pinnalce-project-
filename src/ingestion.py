"""
src/ingestion.py — PDF loading and text cleaning (notebook Section 1).
"""
import os
import re
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger("pipeline")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip null bytes from extracted PDF text."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def discover_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Scan pdf_dir for *.pdf files and build a pdf_sources list automatically.
    document_id is derived from the filename (no extension, lowercased).
    paper_title is the filename stem with underscores replaced by spaces.
    """
    pdf_dir = Path(pdf_dir)
    sources = []
    for i, pdf_path in enumerate(sorted(pdf_dir.glob("*.pdf")), start=1):
        stem = pdf_path.stem
        sources.append({
            "path": str(pdf_path),
            "document_id": f"paper_{i:03d}",
            "paper_title": stem.replace("_", " ").replace("-", " ").title(),
        })
    logger.debug(f"Discovered {len(sources)} PDF(s) in {pdf_dir}")
    return sources


def load_and_clean_pdfs(pdf_sources: list[dict]) -> list[Document]:
    """
    Load each PDF with PyPDFLoader, clean text, attach rich metadata.

    Args:
        pdf_sources: list of dicts with keys 'path', 'document_id', 'paper_title'.

    Returns:
        List of clean Document objects, one per non-empty page.
    """
    clean_documents: list[Document] = []

    for src in pdf_sources:
        logger.info(f"Loading: {src['paper_title']} ({src['path']})")
        loader = PyPDFLoader(src["path"])
        pages = loader.load()

        loaded = 0
        for page in pages:
            cleaned = clean_text(page.page_content)
            if not cleaned:
                continue
            metadata = {
                "document_id":  src["document_id"],
                "paper_title":  src["paper_title"],
                "file_name":    os.path.basename(src["path"]),
                "source":       page.metadata.get("source", src["path"]),
                "page_number":  page.metadata.get("page", 0) + 1,
            }
            clean_documents.append(
                Document(page_content=cleaned, metadata=metadata)
            )
            loaded += 1

        logger.info(
            f"  → {loaded}/{len(pages)} pages loaded from '{src['paper_title']}'"
        )

    logger.info(f"Total clean pages: {len(clean_documents)}")
    return clean_documents
