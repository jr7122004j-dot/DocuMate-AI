"""Vector store lifecycle management for DocuMate-AI.

Provides idempotent creation, loading, deletion and retrieval building on top
of ChromaDB via ``langchain-chroma``.  Collection names are derived from the
document fingerprint produced by :func:`document_processor.process_pdf_bytes`.

Key design points:
    * ``ANONYMIZED_TELEMETRY`` is disabled before importing ``chromadb`` to
      work around a Windows DLL-load failure in the gRPC telemetry backend.
    * Documents are added using their deterministic ``chunk_id`` metadata as
      Chroma document IDs, making repeated ingestion idempotent.
    * Deletion uses ``client.delete_collection()`` rather than
      ``shutil.rmtree`` to avoid Windows file-lock issues.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

# Must be set **before** chromadb is imported anywhere in the process.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from documate_ai.config import Settings, get_settings  # noqa: E402

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collection_name(fingerprint: str) -> str:
    """Derive a Chroma collection name from a document fingerprint.

    Args:
        fingerprint: SHA-256 hex digest of sorted file hashes.

    Returns:
        A collection name of the form ``docs_<first-16-chars>``.
    """
    return f"docs_{fingerprint[:16]}"


def _get_client(settings: Settings) -> chromadb.ClientAPI:
    """Create a PersistentClient pointing at ``settings.chroma_dir``.

    Args:
        settings: Application settings providing the chroma directory.

    Returns:
        A ``chromadb.ClientAPI`` backed by a persistent directory.
    """
    chroma_path = str(settings.chroma_dir)
    return chromadb.PersistentClient(path=chroma_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_or_create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    settings: Settings | None = None,
    fingerprint: str | None = None,
) -> Chroma:
    """Create or reuse a Chroma vector store for the given chunks.

    If a collection already exists and contains the same number of documents,
    re-embedding is skipped.  Otherwise the chunks are upserted using their
    deterministic ``chunk_id`` metadata as Chroma document IDs.

    Args:
        chunks: Chunked ``Document`` objects with ``chunk_id`` in metadata.
        embeddings: Embedding model for vectorisation.
        settings: Application settings (defaults loaded if *None*).
        fingerprint: Document fingerprint used to derive the collection name.
            If *None*, ``chunks[0].metadata["file_hash"]`` is used as a
            fallback.

    Returns:
        A ``Chroma`` vector store wrapper.
    """
    if settings is None:
        settings = get_settings()

    if fingerprint is None:
        fingerprint = chunks[0].metadata.get("file_hash", "default") if chunks else "empty"

    col_name = _collection_name(fingerprint)
    client = _get_client(settings)

    # Probe existing collection count.
    existing_count = 0
    try:
        col = client.get_collection(name=col_name)
        existing_count = col.count()
    except Exception:
        # Collection does not exist yet — that is fine.
        pass

    vs = Chroma(
        client=client,
        collection_name=col_name,
        embedding_function=embeddings,
    )

    if existing_count >= len(chunks) and existing_count > 0:
        logger.info(
            "Collection '%s' already has %d docs (need %d). Skipping re-embedding.",
            col_name,
            existing_count,
            len(chunks),
        )
        return vs

    ids = [c.metadata["chunk_id"] for c in chunks]
    vs.add_documents(chunks, ids=ids)
    logger.info("Added %d chunks to collection '%s'.", len(chunks), col_name)
    return vs


def load_vector_store(
    fingerprint: str,
    embeddings: Embeddings,
    settings: Settings | None = None,
) -> Chroma:
    """Load an existing Chroma collection by its document fingerprint.

    Args:
        fingerprint: Document fingerprint used to derive the collection name.
        embeddings: Embedding model for query vectorisation.
        settings: Application settings (defaults loaded if *None*).

    Returns:
        A ``Chroma`` vector store wrapper.

    Raises:
        ValueError: If the collection does not exist.
    """
    if settings is None:
        settings = get_settings()

    col_name = _collection_name(fingerprint)
    client = _get_client(settings)

    # Verify the collection exists.
    existing_names = [c.name for c in client.list_collections()]
    if col_name not in existing_names:
        raise ValueError(
            f"Collection '{col_name}' does not exist. "
            "Process documents first with get_or_create_vector_store()."
        )

    return Chroma(
        client=client,
        collection_name=col_name,
        embedding_function=embeddings,
    )


def delete_vector_store(
    fingerprint: str,
    settings: Settings | None = None,
) -> None:
    """Delete a Chroma collection identified by its document fingerprint.

    Uses ``client.delete_collection()`` instead of ``shutil.rmtree`` to
    avoid Windows file-lock problems.

    Args:
        fingerprint: Document fingerprint.
        settings: Application settings (defaults loaded if *None*).
    """
    if settings is None:
        settings = get_settings()

    col_name = _collection_name(fingerprint)
    client = _get_client(settings)
    try:
        client.delete_collection(name=col_name)
        logger.info("Deleted collection '%s'.", col_name)
    except Exception:
        logger.debug("Collection '%s' did not exist; nothing to delete.", col_name)


def reset_all_vector_stores(settings: Settings | None = None) -> int:
    """Delete every Chroma collection under the configured directory.

    Args:
        settings: Application settings (defaults loaded if *None*).

    Returns:
        Number of collections deleted.
    """
    if settings is None:
        settings = get_settings()

    client = _get_client(settings)
    collections = client.list_collections()
    for col in collections:
        client.delete_collection(name=col.name)
        logger.info("Deleted collection '%s'.", col.name)
    return len(collections)


def build_retriever(
    store: Chroma,
    settings: Settings | None = None,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """Build a LangChain retriever from a Chroma vector store.

    Args:
        store: The ``Chroma`` vector store to query.
        settings: Application settings providing ``retriever_result_count``.
        search_type: Retrieval strategy — ``"similarity"`` (default) or
            ``"mmr"`` (Maximal Marginal Relevance).

    Returns:
        A ``VectorStoreRetriever`` configured with *k* from settings.
    """
    if settings is None:
        settings = get_settings()

    return store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": settings.retriever_result_count},
    )
