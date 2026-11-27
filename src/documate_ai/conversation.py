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
from documate_ai.question_parser import detect_multiple_questions
from documate_ai.response_processor import (
    build_citations,
    format_response_with_citations,
    render_citations_markdown,
)
from documate_ai.schemas import Citation
from documate_ai.stream_handler import TokenSink
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
        sink: TokenSink | None = None,
    ) -> Iterator[str]:
        """Stream answer tokens for a question.

        Only tokens generated by the QA step are yielded. Condensation and
        retrieval are performed before streaming begins. If a ``sink`` is
        provided, tokens are forwarded to ``sink.on_token()``.

        Args:
            question: The user's question.
            history: Optional conversation history.
            out_result: Optional dictionary populated with final metadata
                (``answer``, ``source_documents``, ``standalone_question``).
            sink: Optional ``TokenSink`` protocol implementer.

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
            if sink is not None:
                sink.on_token(chunk)
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
    sink: TokenSink | None = None,
) -> Iterator[str]:
    """Stream answer tokens using the ConversationalRAG engine.

    Args:
        chain: The ConversationalRAG engine.
        question: The user's question.
        history: Optional ChatHistory.
        out_result: Optional dict populated with metadata upon completion.
        sink: Optional TokenSink receiving tokens in real-time.

    Yields:
        String answer tokens.
    """
    yield from chain.stream_answer(
        question, history=history, out_result=out_result, sink=sink
    )


# ---------------------------------------------------------------------------
# Multi-Question Query Processing
# ---------------------------------------------------------------------------


def _snapshot_history(history: ChatHistory | None) -> ChatHistory | None:
    """Create an isolated snapshot of ChatHistory for sub-question evaluation.

    Args:
        history: The source ChatHistory, or None.

    Returns:
        A new ChatHistory instance with identical messages, or None.
    """
    if history is None:
        return None
    snapshot = ChatHistory(max_turns=history.max_turns)
    snapshot.messages = list(history.messages)
    return snapshot


def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Deduplicate Document objects preserving retrieval order.

    Deduplicates by (source, page, content_prefix) tuple.

    Args:
        docs: List of Document chunks.

    Returns:
        Deduplicated list of Document chunks.
    """
    seen: set[tuple[str, int, str]] = set()
    unique_docs: list[Document] = []
    for doc in docs:
        source = str(doc.metadata.get("source") or "Unknown Document")
        raw_page = doc.metadata.get("page", 1)
        try:
            page = int(raw_page)
        except (ValueError, TypeError):
            page = 1
        key = (source, page, doc.page_content[:150])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def _format_sub_question_heading(question: str) -> str:
    """Format a question string into a markdown heading.

    Ensures a trailing question mark and '### ' markdown heading prefix.

    Args:
        question: Raw question string.

    Returns:
        Markdown heading string (e.g. '### What is the revenue?').
    """
    norm = question.strip()
    if not norm.endswith("?"):
        norm += "?"
    if not norm.startswith("#"):
        norm = f"### {norm}"
    return norm


def process_multi_question_query(
    chain: ConversationalRAG,
    prompt: str,
    history: ChatHistory | None = None,
    sink: TokenSink | None = None,
    settings: Settings | None = None,
    max_sub_questions: int | None = None,
    stream_handler: TokenSink | None = None,
) -> dict[str, Any]:
    """Detect and process multiple questions within a single user prompt.

    If only a single question is detected, it is forwarded through the standard
    chain. When multiple questions are detected, each sub-question is answered
    against the SAME initial chat history snapshot to prevent history pollution.
    Each sub-answer is streamed live under a markdown heading through the TokenSink
    (if provided). Finally, a single combined turn is appended to history.

    Args:
        chain: The ConversationalRAG engine.
        prompt: User input string (may contain multiple questions).
        history: Optional ChatHistory instance.
        sink: Optional TokenSink protocol implementer for live token streaming.
        settings: Application settings.
        max_sub_questions: Optional override for maximum sub-questions to process.
        stream_handler: Backwards-compatible alias for ``sink``.

    Returns:
        Dict compatible with ``ChatResponse`` containing:
            - ``"answer"``: Combined answer string with headings and citations.
            - ``"source_documents"``: Deduplicated list of retrieved Documents.
            - ``"citations"``: Combined, deduplicated Citation list.
            - ``"sub_questions"``: List of processed sub-question strings.
            - ``"sub_answers"``: List of per-sub-question result dicts.
    """
    if sink is None and stream_handler is not None:
        sink = stream_handler

    if settings is None:
        settings = (
            chain.settings
            if hasattr(chain, "settings") and chain.settings is not None
            else get_settings()
        )

    is_multi, questions = detect_multiple_questions(prompt)

    # ------------------------------------------------------------------
    # Single-question path
    # ------------------------------------------------------------------
    if not is_multi or len(questions) <= 1:
        if sink is not None:
            out_res: dict[str, Any] = {}
            for _ in chain.stream_answer(
                prompt, history=history, out_result=out_res, sink=sink
            ):
                pass
            raw_answer = out_res.get("answer", "")
            source_docs = out_res.get("source_documents", [])
            standalone_q = out_res.get("standalone_question", prompt)
        else:
            res = chain.invoke(prompt, history=history)
            raw_answer = res.get("answer", "")
            source_docs = res.get("source_documents", [])
            standalone_q = res.get("standalone_question", prompt)

        final_answer = format_response_with_citations(
            answer=raw_answer,
            source_documents=source_docs,
            question=prompt,
        )
        citations = build_citations(source_docs)

        if history is not None:
            history.add_turn(prompt, final_answer)

        return {
            "answer": final_answer,
            "source_documents": source_docs,
            "citations": citations,
            "sub_questions": [prompt],
            "standalone_question": standalone_q,
            "sub_answers": [
                {
                    "question": prompt,
                    "answer": final_answer,
                    "citations": citations,
                    "source_documents": source_docs,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Multi-question path
    # ------------------------------------------------------------------
    max_q = (
        max_sub_questions
        if max_sub_questions is not None
        else settings.max_sub_questions
    )
    if len(questions) > max_q:
        logger.info(
            "Capping %d detected sub-questions to %d (max_sub_questions limit)",
            len(questions),
            max_q,
        )
        questions = questions[:max_q]

    # Evaluate all sub-questions against the SAME history snapshot
    history_snapshot = _snapshot_history(history)

    combined_parts: list[str] = []
    sub_results: list[dict[str, Any]] = []
    all_source_docs: list[Document] = []

    for i, sub_q in enumerate(questions):
        heading = _format_sub_question_heading(sub_q)
        heading_prefix = f"{heading}\n\n" if i == 0 else f"\n\n{heading}\n\n"

        if sink is not None:
            sink.on_token(heading_prefix)
            sub_out: dict[str, Any] = {}
            for _ in chain.stream_answer(
                sub_q, history=history_snapshot, out_result=sub_out, sink=sink
            ):
                pass
            raw_sub_ans = sub_out.get("answer", "")
            sub_docs = sub_out.get("source_documents", [])
            sub_standalone_q = sub_out.get("standalone_question", sub_q)
            sub_citations = build_citations(sub_docs)
            sub_citations_md = render_citations_markdown(sub_citations)
            if sub_citations_md:
                sink.on_token(sub_citations_md)
        else:
            sub_res = chain.invoke(sub_q, history=history_snapshot)
            raw_sub_ans = sub_res.get("answer", "")
            sub_docs = sub_res.get("source_documents", [])
            sub_standalone_q = sub_res.get("standalone_question", sub_q)
            sub_citations = build_citations(sub_docs)

        sub_ans_formatted = format_response_with_citations(
            answer=raw_sub_ans,
            source_documents=sub_docs,
            question=sub_q,
        )
        combined_parts.append(f"{heading}\n\n{sub_ans_formatted}")
        sub_results.append(
            {
                "question": sub_q,
                "answer": sub_ans_formatted,
                "citations": sub_citations,
                "source_documents": sub_docs,
                "standalone_question": sub_standalone_q,
            }
        )
        all_source_docs.extend(sub_docs)

    combined_answer = "\n\n".join(combined_parts)
    combined_citations = build_citations(all_source_docs)
    deduped_source_docs = _deduplicate_docs(all_source_docs)

    # Append exactly ONE combined turn to history
    if history is not None:
        history.add_turn(prompt, combined_answer)

    return {
        "answer": combined_answer,
        "source_documents": deduped_source_docs,
        "citations": combined_citations,
        "sub_questions": questions,
        "sub_answers": sub_results,
    }
