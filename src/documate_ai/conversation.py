"""Conversational RAG chain construction and query processing.

Wires together the LLM, vector store retriever, chat memory, and
prompt templates into a working conversational retrieval chain.
Also handles multi-question queries by splitting and answering
each sub-question independently.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from documate_ai.config import (
    CONDENSE_QUESTION_TEMPLATE,
    MULTI_QUESTION_STREAM_CHUNK_SIZE,
    QA_PROMPT_TEMPLATE,
    RETRIEVER_RESULT_COUNT,
)
from documate_ai.question_parser import detect_multiple_questions
from documate_ai.response_processor import format_response_with_citations

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel

    from documate_ai.stream_handler import StreamHandler

logger = logging.getLogger(__name__)


def create_conversational_chain(
    vector_store: Chroma,
    llm: BaseChatModel,
) -> ConversationalRetrievalChain:
    """Build a conversational RAG chain from a vector store and LLM.

    The chain combines:
    - A locally hosted LLM (via Ollama) for answer generation
    - A Chroma vector-store retriever for context retrieval
    - A conversation buffer memory for multi-turn context
    - Custom prompt templates for question condensation and
      document-grounded QA

    Args:
        vector_store: A ChromaDB vector store containing embedded
            document chunks.
        llm: The chat language model for generating answers.

    Returns:
        A configured ``ConversationalRetrievalChain`` ready for queries.
    """
    condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    qa_prompt = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_RESULT_COUNT},
        ),
        memory=memory,
        rephrase_question=False,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        output_key="answer",
        verbose=False,
    )
    logger.info("Conversational chain created (retriever k=%d)", RETRIEVER_RESULT_COUNT)
    return chain


def ask_question(
    chain: ConversationalRetrievalChain,
    question: str,
    stream_handler: StreamHandler | None = None,
) -> dict[str, Any]:
    """Send a single question through the RAG chain and return the result.

    Args:
        chain: The conversational retrieval chain.
        question: The user's question string.
        stream_handler: Optional callback handler for streaming tokens
            to the UI in real-time.

    Returns:
        A dict with keys ``"answer"`` (str) and ``"source_documents"``
        (list of Document objects with metadata).
    """
    result = chain(
        {"question": question},
        callbacks=[stream_handler] if stream_handler else None,
    )

    if "source_documents" in result:
        result["answer"] = format_response_with_citations(
            result["answer"],
            result.get("source_documents"),
            question,
        )

    return result


def process_multi_question_query(
    chain: ConversationalRetrievalChain,
    prompt: str,
    stream_handler: StreamHandler | None = None,
) -> dict[str, Any]:
    """Detect and process multiple questions within a single user prompt.

    If only a single question is detected, the prompt is forwarded to
    the chain as-is (with optional streaming). When multiple questions
    are found, each is answered individually; the answers are
    concatenated and streamed in fixed-size chunks for visual consistency.

    Args:
        chain: The conversational retrieval chain.
        prompt: The user's raw input (may contain multiple questions).
        stream_handler: Optional callback handler for streaming.

    Returns:
        A dict with keys ``"answer"`` (str) and ``"source_documents"``
        (list of Document objects).
    """
    is_multi, questions = detect_multiple_questions(prompt)

    # Single question — use the standard path
    if not is_multi or len(questions) <= 1:
        return ask_question(chain, prompt, stream_handler)

    # Multiple questions — answer each independently
    logger.info("Detected %d sub-questions in prompt", len(questions))
    individual_answers: list[tuple[str, str]] = []
    all_source_documents: list = []

    for question in questions:
        result = chain({"question": question}, callbacks=None)
        processed_answer = format_response_with_citations(
            result["answer"],
            result.get("source_documents", []),
            question,
        )
        individual_answers.append((question, processed_answer))
        if "source_documents" in result:
            all_source_documents.extend(result["source_documents"])

    # Build the combined answer text
    combined_parts: list[str] = []
    for question_text, answer_text in individual_answers:
        normalised_question = question_text.strip()
        if not normalised_question.endswith("?"):
            normalised_question += "?"
        combined_parts.append(f"{normalised_question}\n{answer_text}")

    combined_answer = "\n\n".join(combined_parts)

    # Stream the pre-built answer in chunks for visual consistency
    if stream_handler:
        chunk_size = MULTI_QUESTION_STREAM_CHUNK_SIZE
        for offset in range(0, len(combined_answer), chunk_size):
            stream_handler.on_llm_new_token(
                combined_answer[offset:offset + chunk_size],
            )
        stream_handler.finalize()

    return {
        "answer": combined_answer.strip(),
        "source_documents": all_source_documents,
    }
