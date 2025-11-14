from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rag import ColbertRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ColBERT RAG chat for ATS knowledge.")
    parser.add_argument("--index-name", type=str, default="ats_rag_index", help="Name of the ColBERT index to query.")
    parser.add_argument("--metadata-path", type=Path, default=Path("data/chunk_metadata.json"), help="Metadata file emitted during indexing.")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved chunks to use as context.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model identifier for OpenAI Chat Completions.")
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

    conversation: list[dict] = []
    pipeline = ColbertRAGPipeline(
        index_name=args.index_name,
        metadata_path=args.metadata_path,
        k=args.k,
        model=args.model,
    )

    print("Assistant: Hi there! I'm ready to answer general questions and look things up in your docs.")

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

        conversation.append({"role": "user", "content": user_input})
        try:
            result = pipeline.answer(user_input, conversation)
        except Exception as exc:
            print(f"Assistant: I ran into an error: {exc}")
            continue

        answer = result["answer"]
        conversation.append({"role": "assistant", "content": answer})
        print(f"Assistant: {answer}\n")
        if result["used_context"]:
            print(f"(References: {format_sources(result['contexts'])})\n")


if __name__ == "__main__":
    main()
