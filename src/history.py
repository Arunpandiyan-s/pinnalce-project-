"""
src/history.py - Multi-session JSON-backed conversation history.

Session store format (config/chat_history.json):
{
  "active_session": "sess_abc123",
  "sessions": {
    "sess_abc123": {
      "id":         "sess_abc123",
      "title":      "What is BERT?",
      "created_at": "2026-09-06T...",
      "messages":   [ ... ]
    }
  }
}

Backward compat: if the file is a flat list (old format) it is auto-migrated to
a single session titled "Previous Chat" with no data loss.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("pipeline")


def _make_session_id() -> str:
    return "sess_" + uuid.uuid4().hex[:8]


def _new_session(title: str = "New Chat") -> dict:
    return {
        "id":         _make_session_id(),
        "title":      title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages":   [],
    }


def _empty_store() -> dict:
    sess = _new_session("New Chat")
    return {
        "active_session": sess["id"],
        "sessions": {sess["id"]: sess},
    }


def load_sessions(path) -> dict:
    """
    Load the multi-session store from disk.
    - No file -> fresh empty store with one 'New Chat'.
    - Old flat list -> migrates to single session 'Previous Chat'.
    - Valid session store -> returned as-is.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("No history file found - starting fresh.")
        return _empty_store()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read history from {path}: {e} - starting fresh.")
        return _empty_store()

    # Migrate old flat-list format
    if isinstance(data, list):
        logger.info("Migrating old chat_history.json (flat list) -> multi-session format.")
        sess = _new_session("Previous Chat")
        sess["messages"] = data
        store = {
            "active_session": sess["id"],
            "sessions": {sess["id"]: sess},
        }
        save_sessions(store, path)
        return store

    if not isinstance(data, dict) or "sessions" not in data:
        logger.warning("Unrecognised history format - starting fresh.")
        return _empty_store()

    # Ensure active_session points to a real session
    if data.get("active_session") not in data["sessions"]:
        if data["sessions"]:
            data["active_session"] = next(iter(data["sessions"]))
        else:
            fresh = _empty_store()
            data["active_session"] = fresh["active_session"]
            data["sessions"].update(fresh["sessions"])

    logger.debug(f"Loaded {len(data['sessions'])} session(s) from {path.name}")
    return data


def save_sessions(store: dict, path) -> None:
    """Persist the full multi-session store as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    logger.debug(f"Sessions saved ({len(store['sessions'])} sessions) -> {path.name}")


def create_session(store: dict, title: str = "New Chat") -> str:
    """Add a new empty session, make it active. Returns new session id."""
    sess = _new_session(title)
    store["sessions"][sess["id"]] = sess
    store["active_session"] = sess["id"]
    logger.debug(f"Created session {sess['id']}")
    return sess["id"]


def delete_session(store: dict, session_id: str) -> None:
    """Remove a session. If it was active, create a fresh 'New Chat'."""
    store["sessions"].pop(session_id, None)
    if not store["sessions"] or store["active_session"] == session_id:
        create_session(store, "New Chat")
    logger.debug(f"Deleted session {session_id}")


def set_active_session(store: dict, session_id: str) -> None:
    """Switch the active session pointer."""
    if session_id in store["sessions"]:
        store["active_session"] = session_id


def get_session_messages(store: dict, session_id: str) -> list:
    """Return the message list for a session (empty list if not found)."""
    return store["sessions"].get(session_id, {}).get("messages", [])


def auto_title_session(store: dict, session_id: str, first_user_msg: str) -> None:
    """Set the session title from the first user message (max 45 chars)."""
    sess = store["sessions"].get(session_id)
    if sess and sess.get("title") == "New Chat":
        title = first_user_msg.strip()
        sess["title"] = title[:45] + ("..." if len(title) > 45 else "")
        logger.debug(f"Auto-titled session {session_id} -> {sess['title']}")


def append_turn(
    messages: list,
    role: str,
    content: str,
    sources=None,
    web_sources=None,
    source_type=None,
) -> list:
    """
    Append one turn to a message list and return the updated list.
    Same signature as the original - callers do not need to change.
    """
    turn = {
        "role":        role,
        "content":     content,
        "sources":     sources or [],
        "web_sources": web_sources or [],
        "source_type": source_type or (
            "papers" if sources else ("web" if web_sources else "llm")
        ),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    return messages + [turn]


def to_langchain_history(messages: list) -> list:
    """
    Convert JSON history records to LangChain message objects.
    Returns: [HumanMessage(...), AIMessage(...), ...]
    """
    lc_history = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content))
    return lc_history


# Legacy shims - keeps old callers working unchanged
def load_history(path) -> list:
    """Shim: load active session messages only."""
    store = load_sessions(path)
    return get_session_messages(store, store["active_session"])


def save_history(messages: list, path) -> None:
    """Shim: update active session messages only."""
    store = load_sessions(path)
    active = store["active_session"]
    if active in store["sessions"]:
        store["sessions"][active]["messages"] = messages
    save_sessions(store, path)
