"""Explicit conversation history management for DocuMate-AI.

Provides a thread-safe, window-bounded conversation history tracking
HumanMessage and AIMessage pairs without relying on legacy LangChain memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from documate_ai.config import Settings, get_settings


class ChatHistory:
    """Manages multi-turn conversation history bounded by a maximum turn window.

    A 'turn' consists of a user question (HumanMessage) and an assistant
    response (AIMessage).
    """

    def __init__(
        self,
        max_turns: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialise ChatHistory with a sliding window.

        Args:
            max_turns: Maximum number of QA turns to retain. If None,
                defaults to ``settings.chat_history_window``.
            settings: Application settings.
        """
        if settings is None:
            settings = get_settings()

        self.max_turns = (
            max_turns if max_turns is not None else settings.chat_history_window
        )
        self.messages: list[BaseMessage] = []

    def add_turn(self, question: str, answer: str) -> None:
        """Add a complete user-assistant turn and enforce the window.

        Args:
            question: User's question text.
            answer: Assistant's answer text.
        """
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=answer))
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def add_user_message(self, text: str) -> None:
        """Append a human message to history."""
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        """Append an AI message to history."""
        self.messages.append(AIMessage(content=text))

    def clear(self) -> None:
        """Reset conversation history."""
        self.messages.clear()

    @property
    def is_empty(self) -> bool:
        """Return True if history has no messages."""
        return len(self.messages) == 0

    def get_messages(self) -> list[BaseMessage]:
        """Return a copy of the message list."""
        return list(self.messages)

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self.messages)
