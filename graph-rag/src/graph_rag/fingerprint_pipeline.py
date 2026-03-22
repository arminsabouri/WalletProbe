"""
Graph-RAG fingerprint pipeline.

For each heuristic:
  1. Semantic search (UniXcoder via code-graph-rag) -> node IDs
  2. 1-hop CALLS graph expansion -> more node IDs
  3. Fetch source code from disk for all nodes
  4. Ask Claude with the heuristic prompt
  5. Parse and validate the response
"""

from __future__ import annotations

import json
import os
from loguru import logger

from graph_rag.prompts import FEW_SHOT_EXAMPLES, HEURISTICS, SYSTEM_PROMPT

_CYPHER_EXPAND = """
MATCH (n)-[:CALLS]-(neighbor)
WHERE id(n) IN $node_ids
RETURN DISTINCT id(neighbor) AS node_id
LIMIT 25
"""

_CYPHER_SOURCE_LOCATION = """
MATCH (m:Module)-[:DEFINES]->(n)
WHERE id(n) = $node_id
RETURN n.start_line AS start_line, n.end_line AS end_line, m.absolute_path AS path
"""


class GraphFingerprintAnalyzer:
    def __init__(
        self,
        project_name: str,
        anthropic_client,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.project_name = project_name
        self.client = anthropic_client
        self.model = model

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _semantic_search(self, queries: list[str], top_k: int = 8) -> list[int]:
        """Return deduplicated node IDs from semantic search across all queries."""
        from codebase_rag.tools.semantic_search import semantic_code_search

        node_ids: set[int] = set()
        for query in queries:
            results = semantic_code_search(query, top_k=top_k)
            for r in results:
                node_ids.add(r["node_id"])
        return list(node_ids)

    def _expand_with_graph(self, seed_ids: list[int]) -> list[int]:
        """Add 1-hop CALLS neighbours for each seed node. Returns combined unique IDs."""
        if not seed_ids:
            return []

        try:
            from codebase_rag.config import settings
            from codebase_rag.constants import SEMANTIC_BATCH_SIZE
            from codebase_rag.services.graph_service import MemgraphIngestor

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=SEMANTIC_BATCH_SIZE,
            ) as ingestor:
                results = ingestor._execute_query(
                    _CYPHER_EXPAND, {"node_ids": seed_ids}
                )
                neighbour_ids = [r["node_id"] for r in results]

            combined = list(set(seed_ids) | set(neighbour_ids))
            logger.debug(
                f"Graph expansion: {len(seed_ids)} seeds -> {len(combined)} total nodes"
            )
            return combined

        except Exception as e:
            logger.warning(f"Graph expansion failed, using seed nodes only: {e}")
            return seed_ids

    def _fetch_source(self, node_ids: list[int]) -> list[str]:
        """Fetch source code strings for all node IDs using absolute paths, deduplicated."""
        from pathlib import Path

        from codebase_rag.config import settings
        from codebase_rag.constants import SEMANTIC_BATCH_SIZE
        from codebase_rag.services.graph_service import MemgraphIngestor
        from codebase_rag.utils.source_extraction import extract_source_lines

        seen: set[str] = set()
        chunks: list[str] = []

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=SEMANTIC_BATCH_SIZE,
        ) as ingestor:
            for nid in node_ids:
                try:
                    results = ingestor._execute_query(
                        _CYPHER_SOURCE_LOCATION, {"node_id": nid}
                    )
                    if not results:
                        continue
                    row = results[0]
                    path_str = row.get("path")
                    start_line = row.get("start_line")
                    end_line = row.get("end_line")
                    if not path_str or not start_line or not end_line:
                        continue
                    src = extract_source_lines(Path(path_str), start_line, end_line)
                    if src and src not in seen:
                        seen.add(src)
                        chunks.append(src)
                except Exception as e:
                    logger.warning(f"Failed to fetch source for node {nid}: {e}")

        return chunks

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _ask_llm(self, prompt: str, chunks: list[str], max_tokens: int = 16) -> str:
        """Send prompt + code chunks to Claude. Returns the raw response string."""
        if not chunks:
            return "-1"

        full_prompt = f"{prompt}\n\n" + "\n\n---\n\n".join(chunks)

        messages = []
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
        messages.append({"role": "user", "content": full_prompt})

        response = self.client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        text = response.content[0].text.strip()
        logger.debug(f"Claude response: {text!r}")
        return text

    def _parse_binary(self, raw: str, valid_values: tuple | None) -> int:
        try:
            if valid_values is None:
                return int(raw)
            return int(raw) if raw in valid_values else -1
        except ValueError:
            return -1

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_heuristic(self, heuristic: dict) -> int | str:
        key = heuristic["key"]
        logger.info(f"Running heuristic: {key}")

        node_ids = self._semantic_search(heuristic["queries"])
        if not node_ids:
            logger.warning(f"No semantic results for {key}, falling back to -1")
            return -1

        expanded_ids = self._expand_with_graph(node_ids)
        chunks = self._fetch_source(expanded_ids)

        if not chunks:
            logger.warning(f"No source fetched for {key}")
            return -1

        max_tokens = heuristic.get("max_tokens", 100 if heuristic["type"] == "text" else 16)
        raw = self._ask_llm(heuristic["prompt"], chunks, max_tokens=max_tokens)

        if heuristic["type"] == "binary":
            return self._parse_binary(raw, heuristic.get("valid_values"))
        return raw

    def run_all(self) -> dict:
        results: dict[str, int | str] = {}
        for h in HEURISTICS:
            results[h["key"]] = self.run_heuristic(h)
        return results


def run_fingerprint(
    project_name: str,
    output_path: str,
    pretty: bool = False,
    qdrant_db_path: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> None:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()

    if qdrant_db_path:
        os.environ["QDRANT_DB_PATH"] = qdrant_db_path

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    client = anthropic.Anthropic(api_key=api_key)
    analyzer = GraphFingerprintAnalyzer(
        project_name=project_name,
        anthropic_client=client,
        model=model,
    )

    logger.info(f"Starting fingerprint analysis for project: {project_name}")
    try:
        fingerprints = analyzer.run_all()
    finally:
        try:
            from codebase_rag.vector_store import close_qdrant_client
            close_qdrant_client()
        except Exception:
            pass

    output = {"project_name": project_name, "fingerprints": fingerprints}
    indent = 2 if pretty else None
    with open(output_path, "w") as f:
        json.dump(output, f, indent=indent)

    logger.info(f"Fingerprints written to {output_path}")
