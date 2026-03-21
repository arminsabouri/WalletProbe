# Graph RAG

Graph-based retrieval experiments using code-graph-rag + Memgraph.

## Setup

```bash
cd graph-rag
uv sync  # requires Python 3.12+
```

## Usage

Build a code graph for a repository (uses code-graph-rag under the hood; requires Memgraph running):

Start Memgraph (example via Docker):

```bash
docker run -d \
  -p 7687:7687 -p 3000:3000 \
  memgraph/memgraph-mage:latest
```

```bash
uv run python -m graph_rag graph \
  --repo-path ../some-wallet \
  --project-name my-wallet \
  --exclude "tests/**" \
  --clean
```

Flags:

- `--clean`: wipe Memgraph before ingest.
- `--exclude`: extra globs/paths to skip (merged with `.cgrignore` if present).
- `--project-name`: label stored in the graph metadata.
- `--batch-size`: override ingest batch size (defaults to upstream settings).

The command runs `GraphUpdater` + `MemgraphIngestor` from code-graph-rag and leaves the graph in Memgraph (no JSON export).

## Next steps

- Ensure Memgraph is running (`docker-compose up memgraph` in the upstream project, or your own instance).
- Add a `rag` subcommand to query the graph and emit fingerprint answers directly from Memgraph.
