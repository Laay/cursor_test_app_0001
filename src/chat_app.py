from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rag import ColbertChatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ColBERT RAG chat for ATS knowledge.")
    parser.add_argument("--index-name", type=str, default="ats_rag_index", help="Name of the ColBERT index to query.")
    parser.add_argument("--metadata-path", type=Path, default=Path("data/chunk_metadata.json"), help="Metadata file emitted during indexing.")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved chunks to use as context.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model identifier for OpenAI Chat Completions.")
    parser.add_argument("--context-window", type=int, default=6, help="Number of conversation turns to keep in memory.")
    return parser.parse_args()


def format_sources(contexts: List) -> str:
    unique_sources = []
    for ctx in contexts:
        source = ctx.source
        if source not in unique_sources:
            unique_sources.append(source)
    return ", ".join(unique_sources)


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("ATS RAG Chat")
    print("I'm your recruiting assistant. Ask me anything about ATS policies, hiring,")
    print("or your uploaded documents. Type 'exit' to quit.")
    print("=" * 60)

    chatbot = ColbertChatbot(
        index_name=args.index_name,
        metadata_path=args.metadata_path,
        k=args.k,
        model=args.model,
        context_window=args.context_window,
    )

    print(f"Assistant: {chatbot.greet()}")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Talk to you soon!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Thanks for chatting. Goodbye!")
            break

        try:
            response = chatbot.ask(user_input)
        except Exception as exc:
            print(f"Assistant: I ran into an error: {exc}")
            continue

        print(f"Assistant: {response.answer}\n")
        if response.used_context:
            print(f"(References: {format_sources(response.contexts)})\n")


if __name__ == "__main__":
    main()
