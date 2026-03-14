# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto-Fingerprint is a Bitcoin wallet fingerprinting tool. It parses wallet source code, creates vector embeddings of extracted functions, then uses LLM-based semantic search to analyze wallet transaction-construction characteristics (e.g., BIP69 sorting, input types, nLocktime usage, low-R grinding).

## Build & Run Commands

```bash
# Activate environment and install deps
poetry shell
poetry install

# Start Qdrant vector database (required)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z -d qdrant/qdrant

# Create embeddings for a wallet's source code
poetry run python src/auto_fingerprint/main.py embed \
  --dir_to_read test_source_code/sparrow/ \
  --vector_db_uri "localhost:6333"

# Run fingerprint analysis
poetry run python src/auto_fingerprint/main.py fingerprint \
  --dir_to_read test_source_code/sparrow/ \
  --vector_db_uri "localhost:6333"
```

No formal test suite exists. Test wallets are in `test_source_code/` (drongo, sparrow).

## Architecture

**Data flow:** Source code → tree-sitter AST parsing → function extraction → OpenAI embeddings → Qdrant storage → semantic similarity search → LLM fingerprint analysis → metadata stored back in Qdrant.

### Key modules (`src/auto_fingerprint/`)

- **`main.py`** — CLI entry point with two subcommands: `embed` and `fingerprint`. Reads `manifest.json` from the target directory to determine language and wallet name.
- **`source.py`** — `SourceCodeParser` uses tree-sitter to extract functions from Python, Java, and C++ source files.
- **`vector_db.py`** — `QuadrantClient` wraps Qdrant for storing/querying embeddings. Uses a special "metadata" point (UUID derived from SHA256 of "metadata") to store fingerprint results as payload.
- **`response.py`** — `ResponseCollector` runs 10 fingerprinting queries against the vector DB, each with a structured LLM prompt expecting specific response formats (binary 1/0/-1 or enumerated values). Maintains chat history across queries for context.
- **`utils.py`** — Batch embedding creation with token-count filtering (max 8192 tokens) and deduplication.
- **`consts.py`** — Model constants: `text-embedding-3-small` (dim 1536), `gpt-4o-mini`.

### External dependencies

- **Qdrant** (Docker) — vector database, must be running on port 6333
- **OpenAI API** — for embeddings and LLM analysis (key in `.env` as `OPENAI_API_KEY`)

### Manifest format

Each wallet directory needs a `manifest.json`:
```json
{
  "language": "java|python|cpp",
  "version": "x.y.z",
  "source_core_file_suffix": ".java|.py|.cpp",
  "name": "wallet-name"
}
```

Collection names in Qdrant are `{name}-{version}`.
