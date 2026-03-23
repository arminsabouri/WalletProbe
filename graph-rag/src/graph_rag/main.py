import argparse
import os
import sys
from pathlib import Path

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph-based retrieval augmented generation (RAG) playground."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph_parser = subparsers.add_parser(
        "graph", help="Build a code graph using code-graph-rag (Memgraph + JSON export)."
    )
    graph_parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the source repository to analyze.",
    )
    graph_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Memgraph ingest batch size (defaults to code-graph-rag setting).",
    )
    graph_parser.add_argument(
        "--project-name",
        default=None,
        help="Optional project name label stored in the graph.",
    )
    graph_parser.add_argument(
        "--exclude",
        action="append",
        help="Glob/path patterns to exclude (combined with .cgrignore if present).",
    )
    graph_parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the Memgraph database before ingesting.",
    )
    graph_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO).",
    )
    graph_parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["unixcoder", "voyage"],
        help="Embedding provider: 'unixcoder' (local, default) or 'voyage' (Voyage AI API, requires VOYAGE_API_KEY).",
    )

    afp_parser = subparsers.add_parser(
        "agent-fingerprint",
        help="Run heuristic fingerprinting using claude-agent-sdk + code-graph-rag MCP server.",
    )
    afp_parser.add_argument(
        "--project-name",
        required=True,
        help="Project name used during graph build.",
    )
    afp_parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the wallet repository (must match the path used during graph build).",
    )
    afp_parser.add_argument(
        "--output",
        default="fingerprints.json",
        help="Output JSON file path (default: fingerprints.json).",
    )
    afp_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    afp_parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use for fingerprinting (default: claude-sonnet-4-6).",
    )
    afp_parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Path to local Qdrant DB (overrides QDRANT_DB_PATH env var).",
    )
    afp_parser.add_argument(
        "--cgr-bin",
        default=None,
        help="Path to code-graph-rag binary (auto-detected from PATH if omitted).",
    )
    afp_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max heuristics running in parallel (default: 3).",
    )
    afp_parser.add_argument(
        "--save-transcripts",
        action="store_true",
        help="Save full agent transcripts to <output>.transcripts.json.",
    )

    fp_parser = subparsers.add_parser(
        "fingerprint",
        help="Run heuristic fingerprinting against a built graph.",
    )
    fp_parser.add_argument(
        "--project-name",
        required=True,
        help="Project name used during graph build.",
    )
    fp_parser.add_argument(
        "--output",
        default="fingerprints.json",
        help="Output JSON file path (default: fingerprints.json).",
    )
    fp_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    fp_parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Path to local Qdrant DB (overrides QDRANT_DB_PATH env var).",
    )
    fp_parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "graph":
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)
        if args.embedding_provider:
            os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
        from graph_rag.cgr_pipeline import run_graph_build
        repo_path = Path(args.repo_path)
        run_graph_build(
            repo_path=repo_path,
            batch_size=args.batch_size,
            project_name=args.project_name,
            exclude=args.exclude,
            clean=args.clean,
        )
        print("Graph build complete (stored in Memgraph).")

    elif args.command == "agent-fingerprint":
        from graph_rag.agent_pipeline import run_agent_fingerprint
        run_agent_fingerprint(
            project_name=args.project_name,
            repo_path=args.repo_path,
            output_path=args.output,
            pretty=args.pretty,
            qdrant_db_path=args.qdrant_path,
            model=args.model,
            cgr_bin=args.cgr_bin,
            concurrency=args.concurrency,
            save_transcripts=args.save_transcripts,
        )

    elif args.command == "fingerprint":
        from graph_rag.fingerprint_pipeline import run_fingerprint
        run_fingerprint(
            project_name=args.project_name,
            output_path=args.output,
            pretty=args.pretty,
            qdrant_db_path=args.qdrant_path,
            model=args.model,
        )


if __name__ == "__main__":
    main()
