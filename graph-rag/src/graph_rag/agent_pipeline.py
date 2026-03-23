"""
Agent-based fingerprint pipeline using claude-agent-sdk + code-graph-rag MCP server.

For each heuristic, a dedicated Claude agent is spawned with the code-graph-rag
MCP server attached. The agent explores the codebase using query_code_graph /
get_code_snippet / read_file, then returns a structured answer.

All 23 heuristics run in parallel via asyncio.gather().
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

from loguru import logger

from graph_rag.prompts import HEURISTICS, SYSTEM_PROMPT


def _build_format_instructions(heuristic: dict) -> str:
    h_type = heuristic["type"]
    valid = heuristic.get("valid_values")
    if h_type == "binary":
        if valid:
            values = ", ".join(valid)
            return f"respond with ONLY one of these exact values: {values}"
        return "respond with ONLY an integer (e.g. 1, 0, 2, -1)"
    return "respond with ONLY the answer text, no explanation"


def _adapt_prompt(prompt: str) -> str:
    """Rewrite scanner-style prompts (which expect code in the message) for agent use."""
    prompt = prompt.replace(
        "Analyze the following code to", "Find and analyze the relevant code to"
    )
    prompt = prompt.replace("the following code", "the relevant code in the wallet repository")
    return prompt


def _build_agent_prompt(heuristic: dict) -> str:
    hints = ", ".join(heuristic["queries"])
    fmt = _build_format_instructions(heuristic)
    adapted = _adapt_prompt(heuristic["prompt"])
    return (
        "The Bitcoin wallet repository has already been indexed in the code graph database.\n\n"
        "Use the available MCP tools (query_code_graph, get_code_snippet, list_directory, "
        "read_file) to explore the source code and answer this question:\n\n"
        f"{adapted}\n\n"
        f"Search hints — look for code related to: {hints}\n\n"
        f"After exploring the code, {fmt}."
    )


_PARSE_ERROR = object()  # sentinel for unparseable agent response


def _parse_result(raw: str | None, heuristic: dict) -> int | str:
    if not raw:
        return _PARSE_ERROR
    # Claude often explains before giving the answer — take the last non-empty line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    text = lines[-1] if lines else ""
    if heuristic["type"] == "binary":
        valid = heuristic.get("valid_values")
        try:
            val = int(text)
            if valid is None:
                return val
            return val if str(val) in valid else _PARSE_ERROR
        except (ValueError, TypeError):
            return _PARSE_ERROR
    return text


async def _analyze_heuristic(
    heuristic: dict,
    mcp_servers: dict,
    model: str,
    semaphore: asyncio.Semaphore,
    save_transcripts: bool = False,
) -> tuple[int | str, list[dict] | None]:
    import dataclasses
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    key = heuristic["key"]
    prompt = _build_agent_prompt(heuristic)

    options = ClaudeAgentOptions(
        mcp_servers=mcp_servers,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        max_turns=20,
        permission_mode="bypassPermissions",
    )

    result_text: str | None = None
    agent_error: bool = False
    transcript: list[dict] | None = [] if save_transcripts else None
    async with semaphore:
        try:
            async for message in query(prompt=prompt, options=options):
                if transcript is not None:
                    transcript.append(dataclasses.asdict(message))
                if isinstance(message, ResultMessage):
                    result_text = message.result
                    if message.is_error:
                        agent_error = True
                        logger.warning(f"[{key}] agent finished with error: {result_text!r}")
        except Exception as e:
            logger.warning(f"[{key}] agent exception: {e}")
            return -2, transcript

    if agent_error:
        return -2, transcript

    logger.debug(f"[{key}] raw result: {result_text!r}")
    parsed = _parse_result(result_text, heuristic)
    if parsed is _PARSE_ERROR:
        logger.warning(f"[{key}] could not parse agent response: {result_text!r}")
        return -2, transcript
    logger.info(f"[{key}] = {parsed!r}")
    return parsed, transcript


def run_agent_fingerprint(
    project_name: str,
    repo_path: str,
    output_path: str,
    pretty: bool = False,
    qdrant_db_path: str | None = None,
    model: str = "claude-sonnet-4-6",
    cgr_bin: str | None = None,
    concurrency: int = 3,
    save_transcripts: bool = False,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    binary = cgr_bin or shutil.which("code-graph-rag")
    if not binary:
        raise RuntimeError(
            "code-graph-rag binary not found. Pass --cgr-bin or ensure it is on PATH."
        )

    from codebase_rag.config import settings
    from claude_agent_sdk.types import McpStdioServerConfig

    mcp_env: dict[str, str] = {
        "TARGET_REPO_PATH": os.path.abspath(repo_path),
        "CYPHER_PROVIDER": "anthropic",
        "CYPHER_MODEL": "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": api_key,
        "MEMGRAPH_HOST": settings.MEMGRAPH_HOST,
        "MEMGRAPH_PORT": str(settings.MEMGRAPH_PORT),
    }
    if qdrant_db_path:
        mcp_env["QDRANT_DB_PATH"] = qdrant_db_path
    elif os.environ.get("QDRANT_DB_PATH"):
        mcp_env["QDRANT_DB_PATH"] = os.environ["QDRANT_DB_PATH"]

    mcp_servers = {
        "code-graph-rag": McpStdioServerConfig(
            command=binary,
            args=["mcp-server"],
            env=mcp_env,
        )
    }

    async def _run_all() -> tuple[dict, dict]:
        heuristics = HEURISTICS[:3]
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            _analyze_heuristic(h, mcp_servers, model, sem, save_transcripts)
            for h in heuristics
        ]
        pairs = await asyncio.gather(*tasks)
        results = {h["key"]: v for h, (v, _) in zip(heuristics, pairs)}
        transcripts = {h["key"]: t for h, (_, t) in zip(heuristics, pairs)}
        return results, transcripts

    logger.info(f"Starting agent fingerprint analysis for project: {project_name}")
    fingerprints, transcripts = asyncio.run(_run_all())

    output = {"project_name": project_name, "fingerprints": fingerprints}
    indent = 2 if pretty else None
    with open(output_path, "w") as f:
        json.dump(output, f, indent=indent)

    if save_transcripts:
        from pathlib import Path
        transcript_path = Path(output_path).with_suffix(".transcripts.json")
        with open(transcript_path, "w") as f:
            json.dump(transcripts, f, indent=2)
        logger.info(f"Transcripts written to {transcript_path}")

    logger.info(f"Fingerprints written to {output_path}")
