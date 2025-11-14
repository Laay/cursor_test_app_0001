## ATS ColBERT RAG Chat

This project bootstraps a recruitment-focused RAG assistant built on top of a ColBERT dense index. Any files you place in `docs/` (PDF, DOCX, or TXT) are chunked into 300-word windows with 50-word overlap, indexed into a ColBERT collection named `ats_rag_index`, and then used as grounding context when answering user questions. The included chat loop greets the user, handles general ATS conversations, and automatically pulls in up to the three most relevant chunks when a question maps to your documents.

### Project Layout
- `docs/` – add your source files here (a small sample document is provided).
- `data/collection.tsv` – flattened chunk text that ColBERT consumes (auto-generated).
- `data/chunk_metadata.json` – metadata tying ColBERT doc ids back to source files (auto-generated).
- `src/rag/` – ingestion, chunking, indexing, and RAG orchestration modules.
- `src/build_index.py` – CLI utility to build/rebuild the ColBERT index.
- `src/chat_app.py` – terminal chat application that greets the user and answers questions with or without retrieved context.

### Prerequisites
- Python 3.10+ (for the ColBERT + OpenAI stack).
- An `OPENAI_API_KEY` for the LLM that synthesizes final answers.
- Adequate CPU/GPU memory for ColBERT indexing (FAISS runs on CPU by default; adjust requirements if you prefer GPU).

### Setup
```bash
cd /workspace
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # or use a .env file
```

Add your knowledge-base files to `docs/` (PDF/DOCX/TXT). The ingestion pipeline recursively picks up everything with those extensions.

### Build the `ats_rag_index`
```bash
python src/build_index.py \
  --docs-dir docs \
  --index-name ats_rag_index \
  --chunk-size 300 \
  --overlap 50
```
The script prints how many documents/chunks were indexed and writes both the FAISS artifacts (managed by ColBERT) and the metadata JSON under `data/`.

### Run the Chat Application
```bash
python src/chat_app.py \
  --index-name ats_rag_index \
  --metadata-path data/chunk_metadata.json \
  --k 3
```

Behavior:
- The assistant greets you on startup.
- Every turn, it decides whether relevant ColBERT chunks exist. If yes, it cites them; otherwise it falls back to a general ATS answer using the LLM.
- Type `exit` or `quit` (or press Ctrl+C) to end the session.

### Customization Tips
- Change the OpenAI model (`--model` flag) to match what your account supports.
- If you ingest long documents, tweak `--chunk-size`/`--overlap` during indexing.
- The metadata file stores every chunk’s source path and chunk id, making it easy to add UI affordances (links, highlights, etc.) later.

### Troubleshooting
- **“Set OPENAI_API_KEY…”** – Export the key or create a `.env` file at the repo root with the variable defined.
- **“Missing metadata file…”** – Run `src/build_index.py` once before starting the chat app.
- **Dependency errors** – Ensure you are using Python 3.10+ and have system packages required by `faiss-cpu` (on Debian/Ubuntu: `libopenblas-dev`).

With this scaffold you can drop in real policy manuals, interview guides, or compliance PDFs and immediately field ATS-flavored questions via the ColBERT-powered RAG stack.
