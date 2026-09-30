"""PDF document ingestion and chunking pipeline.

Handles the full journey from raw PDF bytes to a searchable ChromaDB
vector store:
    1. Read PDF bytes → extract text page-by-page via PyPDFLoader
    2. Tag each page's metadata with the original filename
    3. Split into overlapping chunks via RecursiveCharacterTextSplitter
    4. Embed all chunks and persist them in ChromaDB
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from documate_ai.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIRECTORY,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def load_pdf(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Extract pages from a PDF and tag metadata with the source filename.

    The PDF bytes are written to a temporary file because ``PyPDFLoader``
    requires a file path. The temp file is deleted after loading.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        filename: Original name of the PDF (used as the ``source`` metadata).

    Returns:
        A list of ``Document`` objects, one per page, each tagged with
        ``metadata["source"]`` set to *filename*.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            logger.warning("Could not delete temp file: %s", temp_path)

    # Tag every page with the original filename
    for document in documents:
        document.metadata["source"] = filename

    logger.info(
        "Loaded %d pages from '%s'", len(documents), filename
    )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split a list of page-level documents into smaller, overlapping chunks.

    Uses ``RecursiveCharacterTextSplitter`` which tries to split at
    natural boundaries (paragraphs → sentences → words) before falling
    back to character-level splitting.

    Args:
        documents: Page-level ``Document`` objects from ``load_pdf``.

    Returns:
        A list of chunk-level ``Document`` objects. Each chunk inherits
        the metadata (source filename, page number) from its parent page.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(
        "Split %d pages into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks


def create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
) -> Chroma:
    """Embed document chunks and store them in a persistent ChromaDB instance.

    Args:
        chunks: Chunk-level ``Document`` objects from ``split_documents``.
        embeddings: The embedding model to vectorise each chunk.

    Returns:
        A ``Chroma`` vector store backed by a persistent on-disk directory.
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    )
    logger.info(
        "Created vector store with %d chunks at '%s'",
        len(chunks),
        CHROMA_PERSIST_DIRECTORY,
    )
    return vector_store


def process_pdfs(
    pdf_files: list[dict],
    embeddings: Embeddings,
) -> Chroma:
    """End-to-end pipeline: ingest multiple PDFs into a single vector store.

    This is the main entry point called by the UI layer. It orchestrates
    ``load_pdf`` → ``split_documents`` → ``create_vector_store``.

    Args:
        pdf_files: List of dicts with keys ``"name"`` (str) and
            ``"bytes"`` (bytes) for each uploaded PDF.
        embeddings: The embedding model to use for vectorisation.

    Returns:
        A ``Chroma`` vector store containing all chunks from all PDFs.

    Raises:
        RuntimeError: If PDF processing fails for any file.
    """
    all_chunks: list[Document] = []

    for pdf_file in pdf_files:
        documents = load_pdf(
            pdf_bytes=pdf_file["bytes"],
            filename=pdf_file["name"],
        )
        chunks = split_documents(documents)
        all_chunks.extend(chunks)

    logger.info(
        "Total: %d chunks from %d PDF(s)", len(all_chunks), len(pdf_files)
    )
    return create_vector_store(all_chunks, embeddings)
