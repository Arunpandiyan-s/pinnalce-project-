"""
config.py — Single source of truth for all tunable constants.
Every module imports from here; no hardcoded paths or names elsewhere.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_DIR             = ROOT_DIR / "data" / "papers"
CHROMA_PERSIST_DIR  = str(ROOT_DIR / "chroma_db")
CONFIG_DIR          = ROOT_DIR / "config"
BEST_CONFIG_PATH    = CONFIG_DIR / "best_config.json"
CHAT_HISTORY_PATH   = CONFIG_DIR / "chat_history.json"
GENERATED_QUESTIONS_PATH = CONFIG_DIR / "generated_questions.json"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "research_papers"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K                = 3
MMR_FETCH_K          = 10
MMR_LAMBDA           = 0.5
SIMILARITY_THRESHOLD = 0.50  # minimum relevance score to consider a paper chunk relevant

# ── Embedding models (key → HuggingFace repo id) ─────────────────────────────
EMBEDDING_MODELS = {
    "bge":   "BAAI/bge-small-en-v1.5",
    "cohere": "embed-english-v3.0",
}

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL   = "gemini-2.5-flash"
LLM_TEMP    = 0

# ── Question generation ───────────────────────────────────────────────────────
NUM_GENERATED_QUESTIONS = 8   # how many Qs the LLM generates per new document
QUESTION_GEN_MAX_PAGES  = 4  # max pages sent to LLM for question generation

# ── Web Search (Tavily) ───────────────────────────────────────────────────────
TAVILY_MAX_RESULTS   = 3      # number of web search results to fetch
TAVILY_SEARCH_DEPTH  = "advanced"  # "basic" or "advanced"
