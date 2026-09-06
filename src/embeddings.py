"""
src/embeddings.py — Embedding model factory (config-driven).
All model names come from config.py; never hardcoded here.
"""
import logging
import time
import cohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
import config

logger = logging.getLogger("pipeline")


def get_embedding_model(key: str) -> HuggingFaceEmbeddings | CohereEmbeddings:
    """
    Return an embedding model instance for the given key.

    Args:
        key: one of the keys in config.EMBEDDING_MODELS (e.g. "bge", "cohere").

    Returns:
        A HuggingFaceEmbeddings instance for HuggingFace-hosted models,
        or a CohereEmbeddings instance for the "cohere" key.
    """
    if key not in config.EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding key '{key}'. "
            f"Valid keys: {list(config.EMBEDDING_MODELS)}"
        )

    model_name = config.EMBEDDING_MODELS[key]
    if key == "cohere":
        return CohereEmbeddings(
            model=model_name,
            client=cohere.Client(),        # reads COHERE_API_KEY from env
            async_client=cohere.AsyncClient(),
        )
    logger.info(f"Loading embedding model '{key}': {model_name}")

    # nomic requires trust_remote_code for its custom pooling layer
    model_kwargs: dict = {"device": "cpu"}
    if key == "nomic":
        model_kwargs["trust_remote_code"] = True

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"  → '{key}' embedding model ready")
    return model


def embed_documents_rate_limited(
    embedding_model,
    texts: list[str],
    batch_size: int = 40,
    cooldown: float = 20,
    inter_batch_delay: float = 1.0,
    max_retries: int = 4,
) -> list[list[float]]:
    """
    Embed *texts* in fixed-size batches with adaptive rate-limit handling.

    Strategy:
      - A small ``inter_batch_delay`` (default 1 s) is used between every batch
        to be polite to the API without wasting time.
      - The large ``cooldown`` sleep is only triggered when a 429 / rate-limit
        error is *actually received*, with exponential back-off on retries
        (cooldown × 2^attempt seconds).

    This is much faster than the old fixed-cooldown approach: for 26 batches the
    old code slept ≈ 26 × 20 s = 8+ minutes unconditionally; now it sleeps only
    ≈ 26 × 1 s = 26 seconds unless the API actually complains.

    Args:
        embedding_model:   Any LangChain embeddings instance.
        texts:             Flat list of strings to embed.
        batch_size:        Number of texts per API call (default 40).
        cooldown:          Seconds to wait after a 429 before retrying
                           (also the base for exponential back-off, default 20).
        inter_batch_delay: Small polite pause between successful batches
                           (default 1 s). Set to 0 to disable.
        max_retries:       Maximum retry attempts per batch on rate-limit errors.

    Returns:
        Combined list of embedding vectors in the same order as *texts*.
    """
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        logger.info(
            f"  embed_documents_rate_limited: batch {batch_idx + 1}/{total_batches} "
            f"({len(batch)} texts)…"
        )

        for attempt in range(max_retries):
            try:
                batch_embeddings = embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                break  # success — move to next batch
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = "429" in exc_str or "rate limit" in exc_str
                if is_rate_limit and attempt < max_retries - 1:
                    wait = cooldown * (2 ** attempt)
                    logger.warning(
                        f"    Rate-limit hit on batch {batch_idx + 1} "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait:.0f}s…"
                    )
                    time.sleep(wait)
                else:
                    raise  # non-rate-limit error, or retries exhausted

        # Small polite pause between batches (skip after the last one)
        if batch_idx < total_batches - 1 and inter_batch_delay > 0:
            logger.debug(f"    Inter-batch delay {inter_batch_delay}s…")
            time.sleep(inter_batch_delay)

    logger.info(
        f"  embed_documents_rate_limited: done — {len(all_embeddings)} embeddings total"
    )
    return all_embeddings
