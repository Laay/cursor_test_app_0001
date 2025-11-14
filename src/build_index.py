from __future__ import annotations

import argparse
from pathlib import Path

from rag import build_colbert_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ATS ColBERT index.")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Directory with source docs.")
    parser.add_argument("--index-name", type=str, default="ats_rag_index", help="Name for the ColBERT index.")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk window size in words.")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in words.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_colbert_index(
        docs_dir=args.docs_dir,
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        f"Built index '{stats['index_name']}' with "
        f"{stats['num_documents']} documents and {stats['num_chunks']} chunks."
    )


if __name__ == "__main__":
    main()
