# Research Paper Answer Bot

A production-style RAG system over research papers, built with LangChain, ChromaDB, and Streamlit.

---

## Architecture

```
project/
├── app.py                    # Streamlit chat UI (thin, imports from src/)
├── config.py                 # ALL constants — never scattered in functions
├── requirements.txt
├── .env.example              # Copy to .env and fill GOOGLE_API_KEY
├── data/papers/              # Drop your PDFs here
├── config/
│   ├── best_config.json      # Written by build_index; read by app.py
│   ├── chat_history.json     # Persisted conversation (JSON)
│   └── generated_questions.json  # LLM-generated eval questions
├── chroma_db/                # Persisted Chroma vector store
├── src/
│   ├── ingestion.py          # PDF loading + text cleaning
│   ├── chunking.py           # Fixed / Recursive / Semantic chunking
│   ├── embeddings.py         # Embedding model factory (bge, nomic)
│   ├── evaluation.py         # Hit@k, MRR, eval question sets
│   ├── question_gen.py       # LLM-based eval question generation
│   ├── vectorstore.py        # Build / load Chroma
│   ├── retrieval.py          # Dense / MMR / Hybrid retrievers
│   ├── web_search.py         # Tavily Web Search integration & tools
│   ├── rag_chain.py          # Single-turn RAG with citations
│   ├── memory_chain.py       # Multi-turn conversational RAG + Web Search
│   ├── history.py            # JSON conversation persistence
│   └── run_logger.py         # Logging bridge (Streamlit + CLI)
├── scripts/
│   ├── build_index.py        # Offline indexing pipeline
│   └── evaluate.py           # Standalone evaluation deliverable
└── notebooks/
    └── experiments.ipynb     # Original exploratory notebook
```

---

## Two-Phase Design

| Phase | When | Command |
|-------|------|---------|
| **Offline indexing** | Once per corpus change | `python scripts/build_index.py` |
| **Online serving** | Always | `streamlit run app.py` |

`app.py` never runs the grid search — it reads `config/best_config.json` and loads the pre-built vector store.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API keys
```bash
cp .env.example .env
# Edit .env and add:
# GOOGLE_API_KEY=your_google_api_key
# TAVILY_API_KEY=your_tavily_api_key
```

### 3. Add research papers
Copy your PDF(s) into `data/papers/`.

### 4a. Build index from CLI (optional — the app does this automatically on upload)
```bash
python scripts/build_index.py
```
This prints three comparison tables:
- 3 chunking × 2 embedding = 6 combinations with Hit@3 / MRR
- Dense vs MMR vs Hybrid retrieval with Hit@3 / MRR
- Writes `config/best_config.json` with the winner + scores

### 4b. Or use the Streamlit UI (recommended)
```bash
streamlit run app.py
```
Upload a PDF in the sidebar → indexing starts automatically with a live log panel showing each step.

### 5. Run standalone evaluation
```bash
python scripts/evaluate.py
```

---

## Features

- **Tavily Web Search Tool**: Toggle live web search in the sidebar to augment paper queries with real-time web results.
- **Dual Citations**: Cites both academic paper pages and clickable web source URLs.
- **Auto-indexing**: Upload PDF → pipeline runs automatically, live logs in sidebar
- **LLM-generated questions**: Gemini reads your paper and creates domain-specific eval Qs
- **Conversational memory**: Multi-turn chat with history-aware retrieval (follow-up questions work)
- **JSON conversation history**: Chat persists across browser refreshes
- **Source citations**: Every answer shows paper title + page number
- **Live run log**: `st.status` panel shows which step is running in real time
- **Architecture overview**: Sidebar shows winning config + scores after indexing

---

## Rubric Coverage

| Criterion | Deliverable |
|-----------|------------|
| Problem Understanding & Data Prep | `src/ingestion.py` — multi-PDF, metadata-rich |
| Vector DB & Embedding | `scripts/build_index.py` printed table (3×2 grid) |
| Retrieval Strategy | `src/retrieval.py` + printed comparison table |
| Web Search Tool | `src/web_search.py` + live Tavily toggle in `app.py` |
| RAG Pipeline | `src/rag_chain.py` — grounded, with citations |
| Stretch Goal 1 (memory) | `src/memory_chain.py` — live in `app.py` |
| Stretch Goal 2 (Streamlit UI) | `app.py` — full chat with paper & web sources |
| Testing & Evaluation | `scripts/evaluate.py` — standalone, demoable |

