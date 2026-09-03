"""
src/web_search.py — Tavily Web Search integration tool & client wrapper.

Provides live web search capabilities, structured result parsing, context formatting,
and LangChain tool integration for the Research Paper Answer Bot.
"""
import os
import logging
from typing import Optional, List, Dict, Any, Literal, cast
import config

logger = logging.getLogger("pipeline")

SearchDepthType = Literal["basic", "advanced", "fast", "ultra-fast"]


def get_tavily_api_key() -> Optional[str]:
    """
    Retrieve Tavily API key from environment variables or Streamlit secrets.
    """
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("TAVILY_API_KEY", "").strip()
        except Exception:
            key = ""
    return key if key else None


def is_tavily_available() -> bool:
    """
    Check if Tavily is configured with a valid-looking API key.
    """
    key = get_tavily_api_key()
    return bool(key and len(key) > 5)


def tavily_search(
    query: str,
    max_results: Optional[int] = None,
    search_depth: Optional[SearchDepthType] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a web search query via Tavily Search API.

    Args:
        query: The search query string.
        max_results: Number of search results (defaults to config.TAVILY_MAX_RESULTS).
        search_depth: "basic" or "advanced" (defaults to config.TAVILY_SEARCH_DEPTH).

    Returns:
        List of dicts: [{"title": str, "url": str, "content": str, "score": float}, ...]
    """
    api_key = get_tavily_api_key()
    if not api_key:
        logger.warning("TAVILY_API_KEY not configured. Skipping web search.")
        return []

    num_results: int = int(max_results if max_results is not None else getattr(config, "TAVILY_MAX_RESULTS", 3))
    raw_depth = str(search_depth or getattr(config, "TAVILY_SEARCH_DEPTH", "advanced"))
    valid_depth: SearchDepthType = cast(
        SearchDepthType,
        raw_depth if raw_depth in ("basic", "advanced", "fast", "ultra-fast") else "advanced"
    )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=num_results,
            search_depth=valid_depth,
            include_answer=False,
        )

        raw_results = response.get("results", [])
        structured = []
        for r in raw_results:
            structured.append({
                "title": r.get("title", "Web Result"),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            })
        logger.info(f"Tavily web search returned {len(structured)} results for: '{query[:50]}'")
        return structured
    except Exception as e:
        logger.error(f"Tavily search failed for query '{query[:50]}': {e}")
        return []


def format_web_context(results: List[Dict[str, Any]]) -> str:
    """
    Format web search results into a clean context block for LLM prompt injection.
    """
    if not results:
        return ""

    blocks = []
    for r in results:
        title = r.get("title", "Web Source")
        url = r.get("url", "")
        content = r.get("content", "").strip()
        header = f"[Web: {title}]" if not url else f"[Web: {title} ({url})]"
        blocks.append(f"{header}\n{content}")

    return "\n\n".join(blocks)


def get_tavily_tool(max_results: Optional[int] = None, search_depth: Optional[SearchDepthType] = None):
    """
    Get a LangChain-compatible Tavily tool instance for agent / tool calling.
    """
    api_key = get_tavily_api_key()
    if not api_key:
        raise ValueError("TAVILY_API_KEY is required to build Tavily tool.")

    num_results: int = int(max_results if max_results is not None else getattr(config, "TAVILY_MAX_RESULTS", 3))
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(
            max_results=num_results,
            tavily_api_key=api_key,
        )
    except Exception as e:
        logger.warning(f"Could not load TavilySearchResults from langchain_community: {e}")
        from langchain_core.tools import tool

        @tool
        def tavily_web_search(query: str) -> str:
            """Search the web for current external information and answers."""
            res = tavily_search(query, max_results=num_results, search_depth=search_depth)
            return format_web_context(res)

        return tavily_web_search

