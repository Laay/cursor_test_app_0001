from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from colbert import Searcher
from dotenv import load_dotenv
from openai import OpenAI

from .index_builder import METADATA_FILE


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class ColbertRAGPipeline:
    """Small helper around ColBERT retrieval + LLM generation."""

    def __init__(
        self,
        index_name: str = "ats_rag_index",
        metadata_path: str | Path = METADATA_FILE,
        k: int = 3,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        load_dotenv()
        self.index_name = index_name
        self.metadata_path = Path(metadata_path)
        self.k = k
        self.model = model
        self.temperature = temperature
        self._load_metadata()
        self.searcher = Searcher(index=index_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set OPENAI_API_KEY in your environment to enable LLM responses."
            )
        self.client = OpenAI()

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata file at {self.metadata_path}. "
                "Run the indexing step first."
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks: List[dict] = metadata.get("chunks", [])

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        ranking = self.searcher.search(query, k=self.k)
        doc_ids = ranking.docids[0]
        scores = ranking.scores[0]

        contexts: list[RetrievedChunk] = []
        for doc_id, score in zip(doc_ids, scores):
            if doc_id == -1:
                continue
            chunk = self.chunks[doc_id]
            contexts.append(
                RetrievedChunk(
                    text=chunk["text"],
                    source=chunk["source"],
                    score=float(score),
                )
            )
        return contexts

    def answer(self, question: str, conversation: List[dict]) -> dict:
        contexts = self.retrieve(question)
        use_context = len(contexts) > 0
        if use_context:
            prompt = self._build_rag_prompt(question, contexts, conversation)
        else:
            prompt = self._build_general_prompt(question, conversation)

        messages = [
            {
                "role": "system",
                "content": "You are ATS RAG, a recruiting assistant who greets users, "
                "answers general ATS questions, and cites retrieved snippets when possible.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        reply = response.choices[0].message.content.strip()
        return {
            "answer": reply,
            "contexts": contexts,
            "used_context": use_context,
        }

    @staticmethod
    def _build_rag_prompt(question: str, contexts: List[RetrievedChunk], conversation: List[dict]) -> str:
        history = _format_history(conversation)
        context_str = "\n\n".join(
            f"Source: {ctx.source}\nExcerpt: {ctx.text}" for ctx in contexts
        )
        return (
            "Use the context chunks from the ATS knowledge base to answer the question. "
            "If information is missing, say you don't know.\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"Context:\n{context_str}\n\n"
            f"User question: {question}"
        )

    @staticmethod
    def _build_general_prompt(question: str, conversation: List[dict]) -> str:
        history = _format_history(conversation)
        return (
            "No domain documents matched this turn. Provide a helpful, general answer "
            "about recruiting, ATS software, or workplace topics.\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"User question: {question}"
        )


def _format_history(conversation: List[dict]) -> str:
    if not conversation:
        return "No prior conversation."
    lines = []
    for turn in conversation[-6:]:
        lines.append(f"{turn['role'].capitalize()}: {turn['content']}")
    return "\n".join(lines)


__all__ = ["ColbertRAGPipeline", "RetrievedChunk"]
