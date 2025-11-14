from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig

from .ingestion import DocumentIngestor, chunk_documents

DATA_DIR = Path("data")
COLLECTION_FILE = DATA_DIR / "collection.tsv"
METADATA_FILE = DATA_DIR / "chunk_metadata.json"


def _persist_collection(chunks: List[dict]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with COLLECTION_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk["text"].replace("\n", " ").strip() + "\n")
    metadata = {
        "chunks": chunks,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return COLLECTION_FILE


def build_colbert_index(
    docs_dir: str | Path = "docs",
    index_name: str = "ats_rag_index",
    chunk_size: int = 300,
    overlap: int = 50,
    checkpoint: str = "colbert-ir/colbertv2.0",
) -> Dict[str, int]:
    """Ingest documents, chunk them, and build a ColBERT index."""

    ingestor = DocumentIngestor(docs_dir)
    documents = ingestor.load()
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No text chunks were produced. Check the source documents.")

    collection_path = _persist_collection(chunks)

    run_config = RunConfig(experiment=index_name, nranks=1)
    colbert_config = ColBERTConfig(
        doc_maxlen=chunk_size,
        query_maxlen=64,
        dim=128,
        similarity="cosine",
        nbits=2,
    )

    with Run().context(run_config):
        indexer = Indexer(checkpoint=checkpoint, config=colbert_config)
        indexer.index(
            name=index_name,
            collection=str(collection_path),
        )

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "index_name": index_name,
    }


__all__ = ["build_colbert_index", "COLLECTION_FILE", "METADATA_FILE"]
