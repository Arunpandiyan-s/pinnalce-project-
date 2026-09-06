"""
app.py - Streamlit Research Paper Answer Bot
Thin UI layer: all business logic lives in src/.

Flow
----
1. Load .env / st.secrets for API keys
2. Load multi-session history from config/chat_history.json
3. Sidebar: session manager + document filter + PDF uploader + config display
4. Load persisted vector store (if index exists) and build chat chain
5. Chat: multi-turn RAG with sources under every answer
"""
import json
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

import config
import src.history as hist
from src.run_logger import attach_streamlit_handler, detach_streamlit_handler

detach_streamlit_handler()

st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="BOOK",
    layout="wide",
)


# =============================================================================
# Cached resource loaders
# =============================================================================
@st.cache_resource
def _load_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it to .env or Streamlit secrets.")
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMP)


@st.cache_resource
def _load_chain(cfg_mtime: float):
    """Load the conversational chain from the persisted vector store.
    cfg_mtime is used as a cache key so the chain reloads when config changes."""
    detach_streamlit_handler()
    from src.embeddings import get_embedding_model
    from src.vectorstore import load_vectorstore
    from src.retrieval import build_dense_retriever, build_mmr_retriever, build_hybrid_retriever
    from src.memory_chain import build_conversational_chain

    with open(config.BEST_CONFIG_PATH) as f:
        cfg = json.load(f)

    emb_model   = get_embedding_model(cfg["embedding_model"])
    vectorstore = load_vectorstore(emb_model, config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME)
    rname = cfg["retrieval_strategy"]
    if rname == "MMR":
        retriever = build_mmr_retriever(
            vectorstore, k=config.TOP_K,
            fetch_k=config.MMR_FETCH_K, lambda_mult=config.MMR_LAMBDA,
        )
    elif "Hybrid" in rname:
        raw = vectorstore.get()
        from langchain_core.documents import Document
        chunks = [Document(page_content=p, metadata=m)
                  for p, m in zip(raw["documents"], raw["metadatas"])]
        retriever = build_hybrid_retriever(chunks, vectorstore, k=config.TOP_K)
    else:
        retriever = build_dense_retriever(vectorstore, k=config.TOP_K)

    llm = _load_llm()
    return build_conversational_chain(retriever, llm), cfg, vectorstore


# =============================================================================
# PDF Upload -> Full Indexing Pipeline
# =============================================================================
def _run_indexing_pipeline(uploaded_files, log_container) -> dict:
    """Save PDFs, run full build pipeline with live logs, generate eval Qs."""
    attach_streamlit_handler(log_container)
    try:
        import logging
        import pandas as pd
        from datetime import datetime, timezone
        logger = logging.getLogger("pipeline")

        from src.ingestion import discover_pdfs, load_and_clean_pdfs
        from src.chunking import get_all_chunking_strategies
        from src.embeddings import get_embedding_model
        from src.evaluation import evaluation_questions, evaluate_chunking_strategy
        from src.vectorstore import build_vectorstore
        from src.retrieval import (
            build_dense_retriever, build_mmr_retriever,
            build_hybrid_retriever, compare_retrieval_strategies,
        )
        from src.question_gen import generate_questions_for_document, save_generated_questions
        from src.evaluation import evaluate_retriever, load_generated_questions

        config.PDF_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Saving uploaded PDF(s)...")
        for uf in uploaded_files:
            dest = config.PDF_DIR / uf.name
            dest.write_bytes(uf.read())
            logger.info(f"  Saved: {uf.name}")

        logger.info("Loading & cleaning PDFs...")
        pdf_sources = discover_pdfs(config.PDF_DIR)
        documents   = load_and_clean_pdfs(pdf_sources)

        logger.info("Loading embedding models...")
        embedding_models = {k: get_embedding_model(k) for k in config.EMBEDDING_MODELS}

        logger.info("Generating chunking strategies (3 x 2 combinations)...")
        all_chunks = get_all_chunking_strategies(
            documents,
            embedding_model=embedding_models["bge"],
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        grid_rows = []
        for strategy_name, chunks in all_chunks.items():
            texts = [c.page_content for c in chunks]
            for emb_key, emb_model in embedding_models.items():
                logger.info(f"  Evaluating: {strategy_name} + {emb_key}...")
                embeddings = emb_model.embed_documents(texts)
                metrics = evaluate_chunking_strategy(
                    chunks, embeddings, evaluation_questions, emb_model, k=config.TOP_K
                )
                grid_rows.append({
                    "Chunking": strategy_name.capitalize(),
                    "Embedding": emb_key, **metrics
                })
                logger.info(f"    Hit@3={metrics['Hit@3']:.3f}  MRR={metrics['MRR']:.3f}")

        grid_df = pd.DataFrame(grid_rows)
        best_row      = grid_df.sort_values(["MRR", "Hit@3"], ascending=False).iloc[0]
        best_chunking = best_row["Chunking"].lower()
        best_emb_key  = best_row["Embedding"]
        logger.info(f"Best combo: {best_chunking} + {best_emb_key}")

        best_chunks    = all_chunks[best_chunking]
        best_emb_model = embedding_models[best_emb_key]

        logger.info("Persisting Chroma vector store...")
        vectorstore = build_vectorstore(
            best_chunks, best_emb_model,
            config.CHROMA_PERSIST_DIR, config.COLLECTION_NAME,
        )

        logger.info("Comparing retrieval strategies...")
        retrievers = {
            "Dense":               build_dense_retriever(vectorstore, k=config.TOP_K),
            "MMR":                 build_mmr_retriever(
                                       vectorstore, k=config.TOP_K,
                                       fetch_k=config.MMR_FETCH_K, lambda_mult=config.MMR_LAMBDA
                                   ),
            "Hybrid (BM25+Dense)": build_hybrid_retriever(best_chunks, vectorstore, k=config.TOP_K),
        }
        retrieval_df = compare_retrieval_strategies(
            retrievers, evaluation_questions, k=config.TOP_K
        )
        best_ret_row  = retrieval_df.sort_values(["MRR", "Hit@3"], ascending=False).iloc[0]
        best_retrieval = best_ret_row["Strategy"]
        logger.info(f"Best retrieval: {best_retrieval}")

        logger.info("Generating evaluation questions with LLM...")
        llm = _load_llm()
        gen_qs = generate_questions_for_document(documents, llm, n=config.NUM_GENERATED_QUESTIONS)
        save_generated_questions(gen_qs, config.GENERATED_QUESTIONS_PATH)

        if gen_qs:
            logger.info("Evaluating with generated questions...")
            best_retriever = retrievers[best_retrieval]
            gen_metrics = evaluate_retriever(best_retriever, gen_qs, k=config.TOP_K)
            logger.info(f"  Hit@3={gen_metrics['Hit@3']:.3f}  MRR={gen_metrics['MRR']:.3f}")
        else:
            gen_metrics = {}

        best_config = {
            "chunking_strategy":  best_chunking,
            "embedding_model":    best_emb_key,
            "retrieval_strategy": best_retrieval,
            "scores": {
                "Hit@3": float(best_ret_row["Hit@3"]),
                "MRR":   float(best_ret_row["MRR"]),
            },
            "generated_q_scores": gen_metrics,
            "grid_results":      grid_df.to_dict(orient="records"),
            "retrieval_results": retrieval_df.to_dict(orient="records"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        config.BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.BEST_CONFIG_PATH, "w") as f:
            json.dump(best_config, f, indent=2)
        logger.info("best_config.json written")
        return best_config
    finally:
        detach_streamlit_handler()


# =============================================================================
# Helper: list all PDF files in data/papers/ with page counts
# =============================================================================
def _get_indexed_documents() -> list[dict]:
    """
    Returns list of dicts: {title: str, filename: str, pages: int}
    Paper titles match what Chroma stores in metadata (paper_title field).
    """
    if not config.PDF_DIR.exists():
        return []
    docs = []
    for pdf_path in sorted(config.PDF_DIR.glob("*.pdf")):
        stem = pdf_path.stem
        title = stem.replace("_", " ").replace("-", " ").title()
        # Count pages quickly using PyPDF
        pages = "?"
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            pages = len(reader.pages)
        except Exception:
            pass
        docs.append({"title": title, "filename": pdf_path.name, "pages": pages})
    return docs


# =============================================================================
# Source badge renderer
# =============================================================================
def _render_source_badge(source_type: str):
    if source_type == "papers":
        st.markdown("Source: Research Papers")
    elif source_type == "web":
        st.markdown("Source: Live Web (Tavily)")
    elif source_type == "hybrid":
        st.markdown("Source: Papers & Live Web")
    elif source_type == "llm":
        st.markdown("Source: General AI Knowledge (Not in uploaded papers)")


# =============================================================================
# Session State Initialisation
# =============================================================================
# Multi-session history
if "session_store" not in st.session_state:
    st.session_state.session_store = hist.load_sessions(config.CHAT_HISTORY_PATH)

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = st.session_state.session_store["active_session"]

# LangChain message history for the active session
if "lc_history" not in st.session_state:
    msgs = hist.get_session_messages(
        st.session_state.session_store, st.session_state.active_session_id
    )
    st.session_state.lc_history = hist.to_langchain_history(msgs)

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = None  # None = all papers


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("Answer Bot")
    st.caption("Research Paper RAG - Powered by Gemini")
    st.divider()

    # ── Chat Sessions ─────────────────────────────────────────────────────────
    st.subheader("Chat Sessions")

    # New Chat button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        new_id = hist.create_session(st.session_state.session_store, "New Chat")
        st.session_state.active_session_id = new_id
        st.session_state.session_store["active_session"] = new_id
        st.session_state.lc_history = []
        hist.save_sessions(st.session_state.session_store, config.CHAT_HISTORY_PATH)
        st.rerun()

    # List existing sessions (newest first)
    sessions = st.session_state.session_store["sessions"]
    session_list = sorted(sessions.values(), key=lambda s: s.get("created_at", ""), reverse=True)

    # Handle pending delete/switch from previous rerun
    if st.session_state.get("_pending_delete_sid"):
        del_sid = st.session_state.pop("_pending_delete_sid")
        if del_sid in st.session_state.session_store["sessions"]:
            hist.delete_session(st.session_state.session_store, del_sid)
            new_active = st.session_state.session_store["active_session"]
            st.session_state.active_session_id = new_active
            msgs = hist.get_session_messages(st.session_state.session_store, new_active)
            st.session_state.lc_history = hist.to_langchain_history(msgs)
            hist.save_sessions(st.session_state.session_store, config.CHAT_HISTORY_PATH)
            st.rerun()

    if st.session_state.get("_pending_switch_sid"):
        sw_sid = st.session_state.pop("_pending_switch_sid")
        if sw_sid in st.session_state.session_store["sessions"] and sw_sid != st.session_state.active_session_id:
            hist.set_active_session(st.session_state.session_store, sw_sid)
            st.session_state.active_session_id = sw_sid
            msgs = hist.get_session_messages(st.session_state.session_store, sw_sid)
            st.session_state.lc_history = hist.to_langchain_history(msgs)
            hist.save_sessions(st.session_state.session_store, config.CHAT_HISTORY_PATH)
            st.rerun()

    for sess in session_list:
        sid = sess["id"]
        title = sess.get("title", "New Chat")
        is_active = sid == st.session_state.active_session_id
        msg_count = len(sess.get("messages", []))

        col_name, col_del = st.columns([5, 1])
        with col_name:
            btn_label = f"\u25cf {title}" if is_active else f"  {title}"
            def _switch(s=sid):
                st.session_state["_pending_switch_sid"] = s
            st.button(
                btn_label,
                key=f"sess_{sid}",
                use_container_width=True,
                help=f"{msg_count // 2} turn(s)",
                on_click=_switch,
            )
        with col_del:
            def _delete(s=sid):
                st.session_state["_pending_delete_sid"] = s
            st.button(
                "\u2715",
                key=f"del_{sid}",
                help="Delete this session",
                on_click=_delete,
            )

    st.divider()

    # ── Indexed Documents (filter) ────────────────────────────────────────────
    indexed_docs = _get_indexed_documents()
    if indexed_docs and config.BEST_CONFIG_PATH.exists():
        st.subheader("Indexed Documents")
        st.caption("Select which papers to search:")

        all_titles = [d["title"] for d in indexed_docs]

        # Default: all selected
        if st.session_state.selected_papers is None:
            st.session_state.selected_papers = all_titles.copy()

        col_all, col_none = st.columns(2)
        with col_all:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_papers = all_titles.copy()
                # Force-update each individual checkbox widget state
                for doc in indexed_docs:
                    st.session_state[f"doc_{doc['filename']}"] = True
                st.rerun()
        with col_none:
            if st.button("Deselect All", use_container_width=True):
                st.session_state.selected_papers = []
                # Force-update each individual checkbox widget state
                for doc in indexed_docs:
                    st.session_state[f"doc_{doc['filename']}"] = False
                st.rerun()

        new_selection = []
        for doc in indexed_docs:
            checked = doc["title"] in st.session_state.selected_papers
            if st.checkbox(
                f"{doc['title']}  ({doc['pages']} pages)",
                value=checked,
                key=f"doc_{doc['filename']}",
            ):
                new_selection.append(doc["title"])
        st.session_state.selected_papers = new_selection

        if not new_selection:
            st.warning("No documents selected. Select at least one to search.")

        st.divider()

    # ── Active Configuration ──────────────────────────────────────────────────
    if config.BEST_CONFIG_PATH.exists():
        with open(config.BEST_CONFIG_PATH) as f:
            saved_cfg = json.load(f)

        st.subheader("Active Configuration")
        st.markdown(f"**Chunking:** `{saved_cfg.get('chunking_strategy', '-')}`")
        st.markdown(f"**Embedding:** `{saved_cfg.get('embedding_model', '-')}`")
        st.markdown(f"**Retrieval:** `{saved_cfg.get('retrieval_strategy', '-')}`")
        scores = saved_cfg.get("scores", {})
        if scores:
            c1, c2 = st.columns(2)
            c1.metric("Hit@3", f"{scores.get('Hit@3', 0):.3f}")
            c2.metric("MRR",   f"{scores.get('MRR',   0):.3f}")

        gen_scores = saved_cfg.get("generated_q_scores", {})
        if gen_scores:
            st.caption("LLM-generated Q evaluation")
            c1, c2 = st.columns(2)
            c1.metric("Hit@3 (gen)", f"{gen_scores.get('Hit@3', 0):.3f}")
            c2.metric("MRR (gen)",   f"{gen_scores.get('MRR',   0):.3f}")
        st.divider()
    else:
        st.info("Upload PDFs below to build the index.")

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.subheader("Upload Research Papers")
    uploaded = st.file_uploader(
        "Drop PDF(s) here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploading auto-rebuilds the index. New PDFs are ADDED to existing ones.",
    )

    if uploaded:
        upload_key = "_".join(sorted(f.name for f in uploaded))
        if upload_key != st.session_state.uploaded_hash:
            st.session_state.uploaded_hash = upload_key
            st.subheader("Build Log")
            log_area = st.empty()

            with st.status("Building index...", expanded=True) as status:
                try:
                    result_cfg = _run_indexing_pipeline(uploaded, log_area)
                    status.update(
                        label=(
                            f"Index ready! "
                            f"Hit@3={result_cfg['scores']['Hit@3']:.3f}  "
                            f"MRR={result_cfg['scores']['MRR']:.3f}"
                        ),
                        state="complete",
                        expanded=False,
                    )
                    # Reset document filter to include all (new docs added)
                    st.session_state.selected_papers = None
                    _load_chain.clear()
                    st.rerun()
                except Exception as e:
                    status.update(label=f"Indexing failed: {e}", state="error")

    # ── Web Search ────────────────────────────────────────────────────────────
    st.subheader("Live Web Search")
    from src.web_search import is_tavily_available
    tavily_ready = is_tavily_available()

    if tavily_ready:
        enable_web_search = st.toggle(
            "Enable Tavily Search",
            value=False,
            help="Search the live web using Tavily alongside research papers.",
        )
        if enable_web_search:
            st.success("Tavily Web Search: Enabled")
        else:
            st.caption("Search Mode: Papers only")
    else:
        enable_web_search = False
        st.caption("Web Search: Disabled")
        st.info("Add TAVILY_API_KEY to your .env to enable live web search.")

    st.divider()
    if st.button("Clear This Chat", use_container_width=True):
        active_id = st.session_state.active_session_id
        if active_id in st.session_state.session_store["sessions"]:
            st.session_state.session_store["sessions"][active_id]["messages"] = []
            st.session_state.session_store["sessions"][active_id]["title"] = "New Chat"
        st.session_state.lc_history = []
        hist.save_sessions(st.session_state.session_store, config.CHAT_HISTORY_PATH)
        st.rerun()


# =============================================================================
# MAIN CHAT AREA
# =============================================================================
st.title("Research Paper Answer Bot")

# Show which session is active
active_sess = st.session_state.session_store["sessions"].get(
    st.session_state.active_session_id, {}
)
active_title = active_sess.get("title", "New Chat")
st.caption(f"Session: **{active_title}** | Ask questions about your uploaded research papers & live web. Sources cited on every answer.")

# Load chain & vector store
chain = None
vectorstore = None
if config.BEST_CONFIG_PATH.exists():
    try:
        cfg_mtime = config.BEST_CONFIG_PATH.stat().st_mtime
        chain, _, vectorstore = _load_chain(cfg_mtime)
    except Exception as e:
        st.error(f"Failed to load chain: {e}")

# Replay saved conversation for the active session
active_messages = hist.get_session_messages(
    st.session_state.session_store, st.session_state.active_session_id
)
for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            src_type = msg.get("source_type")
            if not src_type:
                src_type = "papers" if msg.get("sources") else ("web" if msg.get("web_sources") else "llm")
            _render_source_badge(src_type)

            if msg.get("sources"):
                with st.expander("Paper Sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"* **{s.get('paper_title', 'Unknown')}** - page {s.get('page_number', '?')}")
            if msg.get("web_sources"):
                with st.expander("Web Sources", expanded=False):
                    for w in msg["web_sources"]:
                        url = w.get("url", "")
                        title = w.get("title", "Web Link")
                        if url:
                            st.markdown(f"* [{title}]({url})")
                        else:
                            st.markdown(f"* {title}")
                        if w.get("content"):
                            st.caption(w["content"][:200] + ("..." if len(w["content"]) > 200 else ""))

# Warn if no documents selected
no_docs_selected = (
    st.session_state.selected_papers is not None
    and len(st.session_state.selected_papers) == 0
)

# Chat input
chat_disabled = chain is None or no_docs_selected
if prompt := st.chat_input(
    "Ask about the papers..." if not no_docs_selected else "Select at least one document to search",
    disabled=chat_disabled,
):
    if chain is None:
        st.warning("Upload a PDF first to build the index.")
        st.stop()

    # Determine papers filter (None = all papers)
    selected = st.session_state.selected_papers
    indexed_titles = [d["title"] for d in _get_indexed_documents()]
    # If all papers selected, pass None (no filter = faster)
    paper_filter = None if (selected is None or set(selected) == set(indexed_titles)) else selected

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run chain
    with st.chat_message("assistant"):
        spinner_text = "Searching papers & live web..." if enable_web_search else "Thinking..."
        with st.spinner(spinner_text):
            from src.memory_chain import chat as mem_chat
            llm_instance = _load_llm()
            result = mem_chat(
                chain,
                st.session_state.lc_history,
                prompt,
                use_web_search=enable_web_search,
                llm=llm_instance,
                vectorstore=vectorstore,
                selected_papers=paper_filter,
            )

        answer      = result["answer"]
        sources     = result.get("sources", [])
        web_sources = result.get("web_sources", [])
        source_type = result.get("source_type", "papers")

        st.markdown(answer)
        _render_source_badge(source_type)

        if sources:
            with st.expander("Paper Sources", expanded=True):
                for s in sources:
                    st.caption(f"* **{s.get('paper_title', 'Unknown')}** - page {s.get('page_number', '?')}")

        if web_sources:
            with st.expander("Web Sources", expanded=True):
                for w in web_sources:
                    url = w.get("url", "")
                    title = w.get("title", "Web Link")
                    if url:
                        st.markdown(f"* [{title}]({url})")
                    else:
                        st.markdown(f"* {title}")
                    if w.get("content"):
                        st.caption(w["content"][:250] + ("..." if len(w["content"]) > 250 else ""))

    # Persist turn to active session
    st.session_state.lc_history = result["chat_history"]
    active_id = st.session_state.active_session_id
    current_msgs = hist.get_session_messages(st.session_state.session_store, active_id)

    # Auto-title session from first user message
    hist.auto_title_session(st.session_state.session_store, active_id, prompt)

    updated_msgs = hist.append_turn(current_msgs, "user", prompt)
    updated_msgs = hist.append_turn(
        updated_msgs, "assistant", answer,
        sources=sources, web_sources=web_sources, source_type=source_type,
    )
    st.session_state.session_store["sessions"][active_id]["messages"] = updated_msgs
    hist.save_sessions(st.session_state.session_store, config.CHAT_HISTORY_PATH)
