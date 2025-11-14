from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag_pipeline import ColbertRAGPipeline, RetrievedChunk


@dataclass
class ChatResponse:
    """Container for a chatbot turn."""

    answer: str
    used_context: bool
    contexts: List[RetrievedChunk]


class ColbertChatbot:
    """High-level helper that maintains conversation state for ColBERT RAG chats."""

    def __init__(
        self,
        *,
        index_name: str = "ats_rag_index",
        metadata_path: str = "data/chunk_metadata.json",
        k: int = 3,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        context_window: int = 6,
        greeting: str = "Hi there! I can answer general questions and cite anything I find in your docs.",
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be positive.")
        self.context_window = context_window
        self.greeting = greeting
        self.pipeline = ColbertRAGPipeline(
            index_name=index_name,
            metadata_path=metadata_path,
            k=k,
            model=model,
            temperature=temperature,
        )
        self._conversation: List[dict] = []

    @property
    def conversation(self) -> List[dict]:
        """Return the running conversation history."""
        return list(self._conversation)

    def greet(self) -> str:
        """Return the default greeting message."""
        return self.greeting

    def reset(self) -> None:
        """Clear the stored conversation history."""
        self._conversation.clear()

    def ask(self, question: str) -> ChatResponse:
        """Submit a question, preserving a rolling context window."""
        normalized = question.strip()
        if not normalized:
            raise ValueError("Question cannot be empty.")

        self._conversation.append({"role": "user", "content": normalized})
        history = self._trim_history(include_latest=True)
        result = self.pipeline.answer(normalized, history)
        answer = result["answer"]
        self._conversation.append({"role": "assistant", "content": answer})
        self._trim_history(include_latest=False)
        return ChatResponse(answer=answer, used_context=result["used_context"], contexts=result["contexts"])

    def _trim_history(self, include_latest: bool) -> List[dict]:
        """Keep only the latest window of turns."""
        turn_limit = self.context_window * 2
        if include_latest:
            return self._conversation[-turn_limit:]
        if len(self._conversation) > turn_limit:
            del self._conversation[:-turn_limit]
        return self._conversation[-turn_limit:]


__all__ = ["ColbertChatbot", "ChatResponse"]
