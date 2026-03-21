## Static export

You can export the wallet fingerprints that live in Qdrant as JSON and ship them with a static page (no Flask runtime needed).

1. Make sure Qdrant is running and the scanner has already pushed fingerprints into collections.
2. Export the metadata point from every collection:

```bash
cd ../scanner
poetry run python src/auto_fingerprint/main.py export --vector_db_uri "http://localhost:6333" --output ../server/static/fingerprints.json --pretty
```

3. Open `server/static/index.html` locally or publish the `server/static/` folder with GitHub Pages or any static host. The page reads `fingerprints.json` and renders the comparison table client-side.

This flow works well in CI: run the scanner, call the `export` subcommand, then publish the `server/static/` directory as the build artifact.
