import argparse
from pathlib import Path

from graph_rag.cgr_pipeline import run_graph_build


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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "graph":
        repo_path = Path(args.repo_path)
        run_graph_build(
            repo_path=repo_path,
            batch_size=args.batch_size,
            project_name=args.project_name,
            exclude=args.exclude,
            clean=args.clean,
        )
        print("Graph build complete (stored in Memgraph).")


if __name__ == "__main__":
    main()
