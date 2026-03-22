from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from codebase_rag.config import load_cgrignore_patterns, settings
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.services.graph_service import MemgraphIngestor


def _resolve_excludes(repo_path: Path, cli_excludes: Iterable[str] | None) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """
    Combine .cgrignore entries with CLI --exclude entries to produce exclude/unignore sets.
    Returns (exclude_paths, unignore_paths).
    """
    cgrignore = load_cgrignore_patterns(repo_path)
    exclude_paths = frozenset(cli_excludes) if cli_excludes else frozenset()
    exclude_paths = exclude_paths | (cgrignore.exclude or frozenset())
    unignore_paths = cgrignore.unignore or None
    return exclude_paths or None, unignore_paths


def run_graph_build(
    repo_path: Path,
    batch_size: int | None = None,
    project_name: str | None = None,
    exclude: Iterable[str] | None = None,
    clean: bool = False,
) -> None:
    """
    Build/update the code graph using code-graph-rag's GraphUpdater + MemgraphIngestor
    and keep it in Memgraph (no JSON export).
    """
    repo_path = repo_path.resolve()
    effective_batch = settings.resolve_batch_size(batch_size)

    exclude_paths, unignore_paths = _resolve_excludes(repo_path, exclude)

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=effective_batch,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    ) as ingestor:
        if clean:
            ingestor.clean_database()
            qdrant_path = Path(settings.QDRANT_DB_PATH)
            if qdrant_path.exists():
                shutil.rmtree(qdrant_path)
                logger.info(f"Deleted Qdrant storage at {qdrant_path}")

        ingestor.ensure_constraints()
        parsers, queries = load_parsers()

        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo_path,
            parsers=parsers,
            queries=queries,
            unignore_paths=unignore_paths,
            exclude_paths=exclude_paths,
            project_name=project_name,
        )
        updater.run()
    return None
