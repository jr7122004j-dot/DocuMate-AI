"""Sanity and verification tests for test infrastructure and shared fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pypdf
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from documate_ai.stream_handler import StreamHandler
from test.conftest import ControllableEmbeddings, FakeContainer


def test_fake_embeddings_fixture(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """fake_embeddings should be deterministic and produce 384-dimensional vectors."""
    assert isinstance(fake_embeddings, DeterministicFakeEmbedding)

    vec1 = fake_embeddings.embed_query("DocuMate test query")
    vec2 = fake_embeddings.embed_query("DocuMate test query")
    vec3 = fake_embeddings.embed_query("Different query")

    assert len(vec1) == 384
    assert vec1 == vec2
    assert vec1 != vec3

    docs = fake_embeddings.embed_documents(["doc one", "doc two"])
    assert len(docs) == 2
    assert len(docs[0]) == 384
    assert len(docs[1]) == 384


def test_controllable_embeddings_fixture(
    controllable_embeddings: ControllableEmbeddings,
) -> None:
    """controllable_embeddings should return registered vectors for specific texts."""
    target_vec_a = [1.0] + [0.0] * 383
    target_vec_b = [0.0, 1.0] + [0.0] * 382

    controllable_embeddings.register("query alpha", target_vec_a)
    controllable_embeddings.register("query beta", target_vec_b)

    assert controllable_embeddings.embed_query("query alpha") == target_vec_a
    assert controllable_embeddings.embed_query("query beta") == target_vec_b

    # Unmapped query returns default vector
    unmapped = controllable_embeddings.embed_query("unregistered query")
    assert unmapped == [0.0] * 384

    doc_vectors = controllable_embeddings.embed_documents(["query alpha", "query beta"])
    assert doc_vectors == [target_vec_a, target_vec_b]


def test_fake_chat_model_fixture(fake_chat_model: GenericFakeChatModel) -> None:
    """fake_chat_model should support token streaming and message invocation."""
    response = fake_chat_model.invoke("What is DocuMate-AI?")
    assert isinstance(response, AIMessage)
    assert response.content == "This is a test response from the fake chat model."

    # Streaming token generator
    tokens = [str(chunk.content) for chunk in fake_chat_model.stream("Follow-up")]
    assert "".join(tokens) == "Follow-up response answering the second question."


def test_tmp_directories_fixtures(tmp_cache_dir: Path, tmp_chroma_dir: Path) -> None:
    """tmp_cache_dir and tmp_chroma_dir should be valid existing paths in isolated temp dirs."""
    assert isinstance(tmp_cache_dir, Path)
    assert isinstance(tmp_chroma_dir, Path)
    assert tmp_cache_dir.is_dir()
    assert tmp_chroma_dir.is_dir()
    assert tmp_cache_dir != tmp_chroma_dir

    # Verify write access
    test_file = tmp_cache_dir / "test.txt"
    test_file.write_text("cache test", encoding="utf-8")
    assert test_file.read_text(encoding="utf-8") == "cache test"


def test_pdf_bytes_fixtures(
    sample_pdf_bytes: bytes,
    blank_pdf_bytes: bytes,
    corrupt_pdf_bytes: bytes,
) -> None:
    """PDF fixtures should provide valid multi-page, blank, and corrupt byte streams."""
    # 1. Multi-page sample PDF with known text
    reader = pypdf.PdfReader(io.BytesIO(sample_pdf_bytes))
    assert len(reader.pages) == 3
    page_1_text = reader.pages[0].extract_text()
    page_2_text = reader.pages[1].extract_text()
    page_3_text = reader.pages[2].extract_text()

    assert "DocuMate-AI" in page_1_text
    assert "Semantic Caching" in page_2_text
    assert "Citations" in page_3_text

    # 2. Blank PDF (has pages but no extractable text)
    blank_reader = pypdf.PdfReader(io.BytesIO(blank_pdf_bytes))
    assert len(blank_reader.pages) == 1
    assert blank_reader.pages[0].extract_text().strip() == ""

    # 3. Corrupt PDF bytes raise an exception when parsed
    with pytest.raises(pypdf.errors.PyPdfError):
        corrupt_reader = pypdf.PdfReader(io.BytesIO(corrupt_pdf_bytes))
        _ = corrupt_reader.pages[0].extract_text()


def test_sample_documents_fixture(sample_documents: list[Document]) -> None:
    """sample_documents fixture should provide langchain Documents with source/page metadata."""
    assert len(sample_documents) == 3
    for doc in sample_documents:
        assert isinstance(doc, Document)
        assert len(doc.page_content) > 0
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert isinstance(doc.metadata["page"], int)


def test_fake_container_helper(fake_container: FakeContainer) -> None:
    """FakeContainer should record calls to markdown and expose last_content."""
    assert isinstance(fake_container, FakeContainer)
    assert fake_container.markdown_calls == []
    assert fake_container.last_content is None

    fake_container.markdown("Hello world")
    assert fake_container.markdown_calls == ["Hello world"]
    assert fake_container.last_content == "Hello world"

    fake_container.markdown("Updated content")
    assert len(fake_container.markdown_calls) == 2
    assert fake_container.last_content == "Updated content"


def test_stream_handler_with_fake_container(fake_container: FakeContainer) -> None:
    """StreamHandler should update FakeContainer and finalize cleanly."""
    handler = StreamHandler(container=fake_container)

    handler.on_llm_new_token("Hello")
    handler.on_llm_new_token(" world")
    final_text = handler.finalize()

    assert final_text == "Hello world"
    assert fake_container.last_content == "Hello world"


@pytest.mark.ollama
def test_ollama_auto_skip() -> None:
    """Test marked @pytest.mark.ollama should auto-skip when Ollama server is offline."""
    # This assertion only executes if Ollama is actually online.
    assert True
