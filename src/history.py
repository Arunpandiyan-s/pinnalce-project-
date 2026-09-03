"""
src/history.py — JSON-backed conversation history.

Every chat turn is stored as:
    {"role": "user"|"assistant", "content": str, "sources": [...], "timestamp": str}

On app startup, history is loaded, replayed into the UI, and converted
to LangChain HumanMessage/AIMessage for the memory chain.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("pipeline")


def load_history(path) -> list[dict]:
    """
    Load conversation history from a JSON file.

    Returns:
        List of turn dicts, or [] if the file doesn't exist / is malformed.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded {len(data)} message(s) from {path.name}")
        return data
    except Exception as e:
        logger.warning(f"Could not load history from {path}: {e}")
        return []


def save_history(messages: list[dict], path) -> None:
    """
    Overwrite the history file with the full message list (pretty-printed JSON).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    logger.debug(f"History saved ({len(messages)} messages) → {path.name}")


def append_turn(
    messages: list[dict],
    role: str,
    content: str,
    sources: list[dict] | None = None,
    web_sources: list[dict] | None = None,
    source_type: str | None = None,
) -> list[dict]:
    """
    Append one turn to the message list and return the updated list.

    Args:
        messages: existing message list
        role: "user" or "assistant"
        content: text content of the turn
        sources: list of {"paper_title", "page_number"} dicts (optional)
        web_sources: list of {"title", "url", "content"} dicts (optional)
        source_type: "papers" | "web" | "hybrid" | "llm" (optional)

    Returns:
        New list with the turn appended.
    """
    turn = {
        "role": role,
        "content": content,
        "sources": sources or [],
        "web_sources": web_sources or [],
        "source_type": source_type or ("papers" if sources else ("web" if web_sources else "llm")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return messages + [turn]


def to_langchain_history(messages: list[dict]) -> list:
    """
    Convert JSON history records to LangChain message objects.
    Non-user/assistant roles are skipped.

    Returns:
        [HumanMessage(...), AIMessage(...), ...]
    """
    lc_history = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content))
    return lc_history
