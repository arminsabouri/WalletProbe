# Auto Fingerprint

Create embeddings and collect wallet fingerprint for a given source code.

## Usage

1. Activate poetry environment

```bash
poetry shell poetry env activate
```

2.Install dependencies

```bash
poetry install
```

3.Run qdrant in docker

```bash
docker run -p 6333:6333 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
-d \
qdrant/qdrant
```

4.Create embeddings for some source code

```bash
poetry run python src/auto_fingerprint/main.py embed --dir_to_read test_source_code/electrum/ --vector_db_uri "localhost:6333"
```

5. Create fingerprints

```bash
poetry run python src/auto_fingerprint/main.py fingerprint --dir_to_read test_source_code/electrum/ --vector_db_uri "localhost:6333"
```

### TODO

- Add support for more languages (rust, go, javascript)
- Seed LLM for consistent results
- Explore fine tuning
- Collect more heuristics:
   * Supported output types
   * Fee estimation source
   * Output structure (can support multiple payments)
   * Can RBF
   * Can create OP_RETURN
   * Can create cpfp packages
   * How are inputs sorted?
   * Can create ephemeral anchor outputs
