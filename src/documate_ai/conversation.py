"""Conversational RAG pipeline built with LangChain Expression Language (LCEL).

Wires together the LLM, vector store retriever, conversation history, and
prompt templates into a modern, decoupled conversational retrieval pipeline:
    1. Condense step: If history is present, rewrite follow-up question to be
       standalone; otherwise pass question through unchanged.
    2. Retrieve step: Use standalone question to retrieve relevant chunks.
    3. QA step: Grounded answer generation using the retrieved context and
       the original user question (rephrase_question=False behaviour).
    4. Stream step: Expose stream_answer() yielding ONLY answer tokens.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever

from documate_ai.chat_history import ChatHistory
from documate_ai.config import (
    CONDENSE_QUESTION_SYSTEM_PROMPT,
    QA_HUMAN_PROMPT,
    QA_SYSTEM_PROMPT,
    Settings,
    get_settings,
)
from documate_ai.vector_store import build_retriever

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def format_docs_for_context(docs: list[Document]) -> str:
    """Format retrieved documents with source and page headers for prompt grounding.

    Args:
        docs: List of retrieved Document chunks.

    Returns:
        Formatted context string with ``[source, page N]`` headers.
    """
    if not docs:
        return "No documents retrieved."

    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        parts.append(f"[{source}, page {page}]\n{doc.page_content}")

    return "\n\n".join(parts)


class ConversationalRAG:
    """Conversational RAG engine built on modern LCEL runnables.

    Attributes:
        retriever: Vector store retriever for fetching relevant chunks.
        llm: Chat language model for condensing and QA.
        settings: Application settings.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
        settings: Settings | None = None,
    ) -> None:
        """Initialise ConversationalRAG engine.

        Args:
            retriever: Configured vector store retriever.
            llm: Chat model instance.
            settings: Application settings.
        """
        self.settings = settings if settings is not None else get_settings()
        self.retriever = retriever
        self.llm = llm

        # Condense chain: (history + question) -> standalone question
        self._condense_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONDENSE_QUESTION_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self._condense_chain = self._condense_prompt | self.llm | StrOutputParser()

        # QA chain: (context + question) -> answer
        self._qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", QA_HUMAN_PROMPT),
            ]
        )
        self._qa_chain = self._qa_prompt | self.llm | StrOutputParser()

    def condense_question(
        self,
        question: str,
        history: ChatHistory | None = None,
    ) -> str:
        """Condense a follow-up question using chat history if available.

        If history is empty or None, returns the original question unchanged.

        Args:
            question: The user's input question.
            history: Optional conversation history.

        Returns:
            Standalone question string.
        """
        if history is None or history.is_empty:
            return question

        try:
            condensed = self._condense_chain.invoke(
                {
                    "chat_history": history.get_messages(),
                    "question": question,
                }
            )
            condensed_text = condensed.strip()
            return condensed_text if condensed_text else question
        except Exception as exc:
            logger.warning("Failed to condense question: %s; using original.", exc)
            return question

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query string (standalone question).

        Returns:
            List of matching Document objects.
        """
        return self.retriever.invoke(query)

    def invoke(
        self,
        input_data: dict[str, Any] | str,
        history: ChatHistory | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline for a question.

        Args:
            input_data: Either a question string or a dict with ``"question"``.
            history: Optional conversation history.

        Returns:
            Dict containing ``answer``, ``source_documents``, and
            ``standalone_question``.
        """
        question = (
            input_data["question"] if isinstance(input_data, dict) else str(input_data)
        )

        standalone_question = self.condense_question(question, history)
        source_docs = self.retrieve(standalone_question)
        context_text = format_docs_for_context(source_docs)

        # Grounding: QA prompt receives context and ORIGINAL question
        answer = self._qa_chain.invoke(
            {
                "context": context_text,
                "question": question,
            }
        )

        return {
            "answer": answer,
            "source_documents": source_docs,
            "standalone_question": standalone_question,
        }

    def stream_answer(
        self,
        question: str,
        history: ChatHistory | None = None,
        out_result: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Stream answer tokens for a question.

        Only tokens generated by the QA step are yielded. Condensation and
        retrieval are performed before streaming begins.

        Args:
            question: The user's question.
            history: Optional conversation history.
            out_result: Optional dictionary populated with final metadata
                (``answer``, ``source_documents``, ``standalone_question``).

        Yields:
            Token text chunks as strings.
        """
        standalone_question = self.condense_question(question, history)
        source_docs = self.retrieve(standalone_question)
        context_text = format_docs_for_context(source_docs)

        accumulated_chunks: list[str] = []
        for chunk in self._qa_chain.stream(
            {
                "context": context_text,
                "question": question,
            }
        ):
            accumulated_chunks.append(chunk)
            yield chunk

        full_answer = "".join(accumulated_chunks)
        if out_result is not None:
            out_result.update(
                {
                    "answer": full_answer,
                    "source_documents": source_docs,
                    "standalone_question": standalone_question,
                }
            )


# ---------------------------------------------------------------------------
# Convenience and Backwards-Compatible Helpers
# ---------------------------------------------------------------------------


def create_conversational_chain(
    vector_store: Chroma | VectorStoreRetriever,
    llm: BaseChatModel,
    settings: Settings | None = None,
) -> ConversationalRAG:
    """Build a ConversationalRAG engine from a vector store or retriever and LLM.

    Args:
        vector_store: A Chroma vector store or a configured VectorStoreRetriever.
        llm: Chat language model for answers.
        settings: Application settings.

    Returns:
        Configured ``ConversationalRAG`` instance.
    """
    if settings is None:
        settings = get_settings()

    if isinstance(vector_store, VectorStoreRetriever):
        retriever = vector_store
    else:
        retriever = build_retriever(vector_store, settings)

    return ConversationalRAG(retriever=retriever, llm=llm, settings=settings)


def ask_question(
    chain: ConversationalRAG,
    question: str,
    history: ChatHistory | None = None,
) -> dict[str, Any]:
    """Ask a single question using the ConversationalRAG engine.

    Args:
        chain: The ConversationalRAG engine.
        question: The user's question.
        history: Optional ChatHistory.

    Returns:
        Dict with ``answer``, ``source_documents``, and ``standalone_question``.
    """
    return chain.invoke(question, history=history)


def stream_answer(
    chain: ConversationalRAG,
    question: str,
    history: ChatHistory | None = None,
    out_result: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer tokens using the ConversationalRAG engine.

    Args:
        chain: The ConversationalRAG engine.
        question: The user's question.
        history: Optional ChatHistory.
        out_result: Optional dict populated with metadata upon completion.

    Yields:
        String answer tokens.
    """
    yield from chain.stream_answer(question, history=history, out_result=out_result)
