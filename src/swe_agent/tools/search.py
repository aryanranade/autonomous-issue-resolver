"""Code search tool: search_code (grep-style).

Phase 1 implements regex/substring search over the repo's text files with
sensible ignores. The optional embeddings-based search the brief mentions is
deliberately deferred — it needs an embedding model/API (a dependency and more
rate-limit budget) and grep is enough to validate the agent loop in Phase 2.
The tool's name and result shape won't change if we add a semantic backend.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from swe_agent.llm.base import ToolSpec
from swe_agent.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    require_str,
    resolve_in_root,
    truncate,
)

# Directories never worth searching; they bloat results and burn tokens.
_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".egg-info",
}


def _is_probably_binary(path: Path) -> bool:
    """Cheap binary sniff: a NUL byte in the first chunk."""
    try:
        with path.open("rb") as fh:
            return b"\x00" in fh.read(2048)
    except OSError:
        return True


class SearchCode(Tool):
    """Search file contents for a pattern, returning ``path:line: text`` hits."""

    name = "search_code"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Search the repository's text files for a pattern and return "
                "matching lines as 'path:line: text'. Use this to localize where "
                "a symbol, message, or function is defined or used."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Repository-relative directory or file to "
                        "search under. Defaults to the whole repository.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat pattern as a regex. Default false "
                        "(literal substring).",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive match. Default false.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matching lines to return. Default 100.",
                    },
                },
                "required": ["pattern"],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        pattern = require_str(args, "pattern")
        rel = args.get("path", ".")
        if not isinstance(rel, str):
            return ToolResult.error("argument 'path' must be a string")
        use_regex = bool(args.get("regex", False))
        ignore_case = bool(args.get("ignore_case", False))
        max_results = args.get("max_results", 100)
        if not isinstance(max_results, int) or max_results <= 0:
            return ToolResult.error("max_results must be a positive integer")

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern if use_regex else re.escape(pattern), flags)
        except re.error as exc:
            return ToolResult.error(f"invalid regex: {exc}")

        # Resolve the root too: resolve_in_root returns real (symlink-resolved)
        # paths, so relative_to must use the resolved root or it can mismatch
        # (e.g. macOS /var -> /private/var).
        root = ctx.root.resolve()
        base = resolve_in_root(ctx.root, rel)
        if not base.exists():
            return ToolResult.error(f"path not found: {rel}")

        files = [base] if base.is_file() else self._walk(base)
        hits: list[str] = []
        for file in files:
            if _is_probably_binary(file):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relname = file.relative_to(root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{relname}:{lineno}: {line.strip()}")
                    if len(hits) >= max_results:
                        body = "\n".join(hits)
                        return ToolResult.success(
                            truncate(
                                f"{body}\n... [stopped at max_results={max_results}]",
                                ctx.max_output_chars,
                            )
                        )

        if not hits:
            return ToolResult.success(f"no matches for {pattern!r}")
        return ToolResult.success(truncate("\n".join(hits), ctx.max_output_chars))

    @staticmethod
    def _walk(base: Path) -> list[Path]:
        """Yield text-candidate files under ``base``, skipping ignored dirs."""
        out: list[Path] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            out.append(path)
        return sorted(out)
