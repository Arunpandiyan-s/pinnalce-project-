"""
src/question_gen.py — LLM-based evaluation question generation.

When a new PDF is uploaded, this module asks the Gemini LLM to read
the document text and produce N question–expected_page pairs in JSON.
These are persisted to config/generated_questions.json and merged into
the evaluation pipeline automatically.
"""
import json
import logging
import re
from pathlib import Path
from langchain_core.documents import Document

import config

logger = logging.getLogger("pipeline")

_GEN_PROMPT = """You are an expert at creating evaluation questions for information retrieval systems.

Given the following research paper text (with page numbers), generate exactly {n} factual questions \
that can be answered from the text. For each question, also record the page number where the answer \
can most directly be found.

Rules:
- Questions must be specific and answerable from the text alone.
- Do NOT generate yes/no questions.
- Return ONLY a valid JSON array — no explanations, no markdown fences.
- Format: [{{"question": "...", "expected_page": <integer>}}, ...]

Paper text:
{text}
"""


def _build_document_text(documents: list[Document], max_pages: int) -> str:
    """Build a labelled text block from the first max_pages pages."""
    lines = []
    seen_pages: set = set()
    for doc in documents:
        page = doc.metadata.get("page_number", 0)
        if page in seen_pages or page > max_pages:
            continue
        seen_pages.add(page)
        lines.append(f"[Page {page}]\n{doc.page_content[:800]}")
    return "\n\n".join(lines)


def generate_questions_for_document(
    documents: list[Document],
    llm,
    n: int | None = None,
) -> list[dict]:
    """
    Generate n evaluation question–page pairs using the LLM.

    Args:
        documents: cleaned Document list for the uploaded paper.
        llm: a LangChain ChatModel (e.g. ChatGoogleGenerativeAI).
        n: number of questions to generate (defaults to config.NUM_GENERATED_QUESTIONS).

    Returns:
        List of {"question": str, "expected_page": int} dicts.
        Returns [] on parse failure.
    """
    n = n or config.NUM_GENERATED_QUESTIONS
    max_pages = config.QUESTION_GEN_MAX_PAGES

    logger.info(f"Generating {n} evaluation questions with LLM…")
    text = _build_document_text(documents, max_pages)

    prompt = _GEN_PROMPT.format(n=n, text=text)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

        questions = json.loads(raw)
        # Validate shape
        valid = [
            q for q in questions
            if isinstance(q, dict)
            and "question" in q
            and "expected_page" in q
            and isinstance(q["expected_page"], int)
        ]
        logger.info(f"Generated {len(valid)} valid question(s)")
        return valid
    except Exception as e:
        logger.warning(f"Question generation failed: {e}")
        return []


def save_generated_questions(questions: list[dict], path) -> None:
    """
    Append new questions to the generated_questions.json file.
    Deduplicates by question text before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing_qs = {q["question"] for q in existing}
    new_qs = [q for q in questions if q["question"] not in existing_qs]
    merged = existing + new_qs

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Saved {len(new_qs)} new question(s) to {path.name} "
        f"(total: {len(merged)})"
    )
