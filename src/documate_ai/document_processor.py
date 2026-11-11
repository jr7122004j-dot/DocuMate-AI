"""PDF document ingestion and chunking pipeline.

Handles the journey from raw PDF bytes to normalised, chunked LangChain Documents:
    1. Read PDF bytes in memory via ``pypdf.PdfReader`` (no temp-file disk writes).
    2. Enforce limits: file size (max_upload_size_mb) and page count (max_pdf_pages).
    3. Detect empty or scanned (image-only) PDFs and skip with warnings.
    4. Normalise text: strip control characters, collapse excessive whitespace.
    5. Split into overlapping chunks with 1-based page numbers and deterministic chunk IDs.
    6. Return chunks along with an ``IngestSummary``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documate_ai.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Settings,
    get_settings,
)
from documate_ai.errors import DocumentProcessingError

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class UploadedPdf:
    """Represents an uploaded PDF file with its filename and raw byte content."""

    name: str
    data: bytes


@dataclass
class IngestSummary:
    """Summary of the PDF ingestion and chunking process."""

    total_files: int
    processed_files: list[str] = field(default_factory=list)
    skipped_files: dict[str, str] = field(default_factory=dict)
    total_pages: int = 0
    total_chunks: int = 0
    document_fingerprint: str = ""


def normalize_text(text: str) -> str:
    """Normalise extracted text by stripping control chars and collapsing whitespace."""
    if not text:
        return ""
    # Strip null characters and non-printable control characters except \t and \n
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple horizontal whitespace characters into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Strip whitespace surrounding line breaks
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def load_pdf_pages(
    pdf: UploadedPdf,
    settings: Settings,
) -> tuple[list[Document], str]:
    """Extract text from a PDF's pages using in-memory pypdf.PdfReader.

    Args:
        pdf: The uploaded PDF to process.
        settings: Application settings for limits.

    Returns:
        Tuple of (list of page Documents, SHA-256 file hash).

    Raises:
        DocumentProcessingError: If the PDF is corrupted, encrypted, or exceeds limits.
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(pdf.data) > max_bytes:
        raise DocumentProcessingError(
            f"File '{pdf.name}' exceeds maximum upload size of "
            f"{settings.max_upload_size_mb} MB ({len(pdf.data)} bytes).",
            filename=pdf.name,
        )

    file_hash = hashlib.sha256(pdf.data).hexdigest()

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf.data))
        if reader.is_encrypted:
            raise DocumentProcessingError(
                f"File '{pdf.name}' is encrypted or password-protected.",
                filename=pdf.name,
            )
        page_count = len(reader.pages)
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to read PDF '{pdf.name}': {exc}",
            filename=pdf.name,
        ) from exc

    if page_count > settings.max_pdf_pages:
        raise DocumentProcessingError(
            f"File '{pdf.name}' has {page_count} pages, which exceeds the maximum limit "
            f"of {settings.max_pdf_pages} pages.",
            filename=pdf.name,
        )

    page_docs: list[Document] = []
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1  # 1-based page numbering
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Failed to extract text from page %d of '%s': %s",
                page_num,
                pdf.name,
                exc,
            )
            raw_text = ""

        clean_text = normalize_text(raw_text)
        if clean_text:
            page_docs.append(
                Document(
                    page_content=clean_text,
                    metadata={
                        "source": pdf.name,
                        "page": page_num,
                        "file_hash": file_hash,
                    },
                )
            )

    return page_docs, file_hash


def process_pdf_bytes(
    uploaded_files: list[UploadedPdf] | list[dict[str, Any]],
    settings: Settings | None = None,
) -> tuple[list[Document], IngestSummary]:
    """Process uploaded PDF files into normalised, chunked Documents with metadata.

    Args:
        uploaded_files: List of ``UploadedPdf`` instances (or dicts with name/bytes) to process.
        settings: Application settings. If None, default settings are loaded.

    Returns:
        Tuple containing:
            - List of chunk-level ``Document`` objects.
            - An ``IngestSummary`` detailing processed files, skipped files, and counts.

    Raises:
        DocumentProcessingError: If all files are empty/unreadable or if any
            file is corrupt/invalid.
    """
    if settings is None:
        settings = get_settings()

    if not uploaded_files:
        raise DocumentProcessingError("No PDF files provided for processing.")

    # Convert any dict inputs into UploadedPdf instances
    validated_files: list[UploadedPdf] = []
    for item in uploaded_files:
        if isinstance(item, UploadedPdf):
            validated_files.append(item)
        elif isinstance(item, dict):
            name = str(item.get("name", "document.pdf"))
            data = item.get("data") or item.get("bytes") or item.get("content") or b""
            validated_files.append(UploadedPdf(name=name, data=data))
        else:
            raise DocumentProcessingError(f"Unsupported file input type: {type(item)}")

    all_chunks: list[Document] = []
    processed_files: list[str] = []
    skipped_files: dict[str, str] = {}
    processed_file_hashes: list[str] = []
    total_pages = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    for pdf in validated_files:
        page_docs, file_hash = load_pdf_pages(pdf, settings)

        if not page_docs:
            logger.warning(
                "PDF '%s' contains no extractable text (empty or scanned image-only). Skipping.",
                pdf.name,
            )
            skipped_files[pdf.name] = (
                "No extractable text found (empty or scanned image-only PDF)"
            )
            continue

        processed_files.append(pdf.name)
        processed_file_hashes.append(file_hash)
        total_pages += len(page_docs)

        file_chunk_index = 0
        for page_doc in page_docs:
            page_num = page_doc.metadata["page"]
            chunks = text_splitter.split_documents([page_doc])
            for chunk in chunks:
                chunk_id = hashlib.sha256(
                    f"{file_hash}:{page_num}:{file_chunk_index}".encode("utf-8")
                ).hexdigest()
                chunk.metadata["source"] = pdf.name
                chunk.metadata["page"] = page_num
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["chunk_index"] = file_chunk_index
                chunk.metadata["chunk_id"] = chunk_id
                if hasattr(chunk, "id"):
                    chunk.id = chunk_id

                all_chunks.append(chunk)
                file_chunk_index += 1

    if not all_chunks:
        raise DocumentProcessingError(
            "No extractable text found in any of the uploaded PDF files."
        )

    document_fingerprint = hashlib.sha256(
        "".join(sorted(processed_file_hashes)).encode("utf-8")
    ).hexdigest()

    summary = IngestSummary(
        total_files=len(validated_files),
        processed_files=processed_files,
        skipped_files=skipped_files,
        total_pages=total_pages,
        total_chunks=len(all_chunks),
        document_fingerprint=document_fingerprint,
    )

    logger.info(
        "Ingested %d chunks from %d files (%d pages total, fingerprint=%s)",
        len(all_chunks),
        len(processed_files),
        total_pages,
        document_fingerprint[:8],
    )
    return all_chunks, summary


# ---------------------------------------------------------------------------
# Backwards-compatible convenience helpers
# ---------------------------------------------------------------------------


def load_pdf(
    pdf_bytes: bytes,
    filename: str,
    settings: Settings | None = None,
) -> list[Document]:
    """Extract pages from a PDF and tag metadata with source and 1-based page numbers."""
    if settings is None:
        settings = get_settings()
    pdf = UploadedPdf(name=filename, data=pdf_bytes)
    page_docs, _ = load_pdf_pages(pdf, settings)
    return page_docs


def split_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Split a list of page-level documents into smaller, overlapping chunks."""
    chunk_size = settings.chunk_size if settings else CHUNK_SIZE
    chunk_overlap = settings.chunk_overlap if settings else CHUNK_OVERLAP
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
) -> Any:
    """Embed document chunks and store them in ChromaDB."""
    from langchain_chroma import Chroma

    directory = persist_directory or str(CHROMA_PERSIST_DIRECTORY)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=directory,
    )


def process_pdfs(
    pdf_files: list[dict[str, Any]],
    embeddings: Embeddings,
    settings: Settings | None = None,
) -> Any:
    """End-to-end pipeline: ingest multiple PDFs into a single Chroma vector store."""
    chunks, _ = process_pdf_bytes(pdf_files, settings)
    return create_vector_store(chunks, embeddings)
