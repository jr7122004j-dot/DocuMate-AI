"""Unit tests for the PDF document processor and chunking pipeline."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from documate_ai.config import Settings
from documate_ai.document_processor import (
    UploadedPdf,
    normalize_text,
    process_pdf_bytes,
)
from documate_ai.errors import DocumentProcessingError


def test_1_based_page_numbering(sample_pdf_bytes: bytes) -> None:
    """Chunks should contain 1-based page numbers in metadata."""
    pdf = UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)
    chunks, summary = process_pdf_bytes([pdf])

    assert len(chunks) > 0
    pages = [c.metadata["page"] for c in chunks]
    assert 0 not in pages
    assert 1 in pages
    assert pages == [1, 2, 3]
    assert summary.total_pages == 3
    assert summary.processed_files == ["sample.pdf"]


def test_deterministic_chunk_id_generation(sample_pdf_bytes: bytes) -> None:
    """chunk_id should be deterministic based on file_hash, page, and chunk_index."""
    pdf = UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)
    chunks1, summary1 = process_pdf_bytes([pdf])
    chunks2, summary2 = process_pdf_bytes([pdf])

    assert len(chunks1) == len(chunks2)
    file_hash = hashlib.sha256(sample_pdf_bytes).hexdigest()

    for idx, chunk in enumerate(chunks1):
        page = chunk.metadata["page"]
        chunk_idx = chunk.metadata["chunk_index"]
        expected_chunk_id = hashlib.sha256(
            f"{file_hash}:{page}:{chunk_idx}".encode("utf-8")
        ).hexdigest()

        assert chunk.metadata["chunk_id"] == expected_chunk_id
        assert chunk.metadata["file_hash"] == file_hash
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["chunk_id"] == chunks2[idx].metadata["chunk_id"]

    assert summary1.document_fingerprint == summary2.document_fingerprint


def test_blank_pdf_skipped_and_error_if_only_blank(
    blank_pdf_bytes: bytes,
    sample_pdf_bytes: bytes,
) -> None:
    """Blank/scanned PDFs should be skipped; error raised if all files are blank."""
    blank_pdf = UploadedPdf(name="blank.pdf", data=blank_pdf_bytes)
    with pytest.raises(DocumentProcessingError, match="No extractable text"):
        process_pdf_bytes([blank_pdf])

    sample_pdf = UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)
    chunks, summary = process_pdf_bytes([sample_pdf, blank_pdf])

    assert "sample.pdf" in summary.processed_files
    assert "blank.pdf" in summary.skipped_files
    assert summary.total_files == 2
    assert len(chunks) > 0


def test_corrupted_pdf_handling(corrupt_pdf_bytes: bytes) -> None:
    """Corrupted PDF should raise DocumentProcessingError with filename."""
    pdf = UploadedPdf(name="bad_corrupt.pdf", data=corrupt_pdf_bytes)
    with pytest.raises(DocumentProcessingError) as exc_info:
        process_pdf_bytes([pdf])

    assert exc_info.value.filename == "bad_corrupt.pdf"
    assert "bad_corrupt.pdf" in str(exc_info.value)


def test_max_pdf_pages_restriction(sample_pdf_bytes: bytes) -> None:
    """PDFs exceeding max_pdf_pages should raise DocumentProcessingError."""
    settings = Settings(max_pdf_pages=2)
    pdf = UploadedPdf(name="too_many_pages.pdf", data=sample_pdf_bytes)

    with pytest.raises(DocumentProcessingError) as exc_info:
        process_pdf_bytes([pdf], settings=settings)

    assert exc_info.value.filename == "too_many_pages.pdf"
    assert "exceeds the maximum limit" in str(exc_info.value)


def test_max_upload_size_restriction(sample_pdf_bytes: bytes) -> None:
    """PDFs exceeding max_upload_size_mb should raise DocumentProcessingError."""
    settings = Settings(max_upload_size_mb=0)
    pdf = UploadedPdf(name="too_big.pdf", data=sample_pdf_bytes)

    with pytest.raises(DocumentProcessingError) as exc_info:
        process_pdf_bytes([pdf], settings=settings)

    assert exc_info.value.filename == "too_big.pdf"
    assert "exceeds maximum upload size" in str(exc_info.value)


def test_encrypted_pdf_handling() -> None:
    """Encrypted/password-protected PDFs should raise DocumentProcessingError."""
    mock_reader = MagicMock()
    mock_reader.is_encrypted = True

    with patch("pypdf.PdfReader", return_value=mock_reader):
        pdf = UploadedPdf(name="encrypted.pdf", data=b"%PDF-1.4 mock")
        with pytest.raises(DocumentProcessingError) as exc_info:
            process_pdf_bytes([pdf])

        assert exc_info.value.filename == "encrypted.pdf"
        assert "encrypted" in str(exc_info.value).lower()


def test_normalize_text() -> None:
    """normalize_text should strip nulls, control chars, and collapse whitespace."""
    raw = "Hello\x00 World!\x01\x02  \t  This   is \n\n\n\n a test.\n  Line 2.  "
    clean = normalize_text(raw)
    assert "\x00" not in clean
    assert "\x01" not in clean
    assert "  " not in clean
    assert clean == "Hello World! This is\n\na test.\nLine 2."


def test_empty_file_list_raises() -> None:
    """Processing empty list of files should raise DocumentProcessingError."""
    with pytest.raises(DocumentProcessingError, match="No PDF files"):
        process_pdf_bytes([])


def test_dict_input_compatibility(sample_pdf_bytes: bytes) -> None:
    """process_pdf_bytes should accept dict format for backwards compatibility."""
    dict_file = {"name": "sample_dict.pdf", "bytes": sample_pdf_bytes}
    chunks, summary = process_pdf_bytes([dict_file])

    assert len(chunks) > 0
    assert summary.processed_files == ["sample_dict.pdf"]


def test_load_pdf_and_split_documents_helpers(sample_pdf_bytes: bytes) -> None:
    """Legacy helper functions load_pdf and split_documents should work as expected."""
    from documate_ai.document_processor import load_pdf, split_documents

    docs = load_pdf(sample_pdf_bytes, "sample.pdf")
    assert len(docs) == 3
    assert docs[0].metadata["page"] == 1

    chunks = split_documents(docs)
    assert len(chunks) >= 3

