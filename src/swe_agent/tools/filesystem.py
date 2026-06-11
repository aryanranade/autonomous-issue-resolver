"""Filesystem tools: read_file, list_dir, edit_file.

All paths are repo-relative and contained to the root via ``resolve_in_root``.
"""

from __future__ import annotations

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


class ReadFile(Tool):
    """Read a text file, optionally a line range, with line numbers."""

    name = "read_file"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read a text file from the repository and return its contents "
                "with 1-based line numbers. Optionally restrict to a line range "
                "to avoid pulling huge files into context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive).",
                    },
                },
                "required": ["path"],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = resolve_in_root(ctx.root, require_str(args, "path"))
        if not path.exists():
            return ToolResult.error(f"file not found: {args['path']}")
        if path.is_dir():
            return ToolResult.error(f"{args['path']} is a directory, not a file")

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        start = args.get("start_line", 1)
        end = args.get("end_line", len(lines))
        if not isinstance(start, int) or not isinstance(end, int):
            return ToolResult.error("start_line and end_line must be integers")
        start = max(1, start)
        end = min(len(lines), end)
        if start > end:
            return ToolResult.error(
                f"empty range: start_line {start} > end_line {end} "
                f"(file has {len(lines)} lines)"
            )

        numbered = "\n".join(
            f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1)
        )
        return ToolResult.success(truncate(numbered, ctx.max_output_chars))


class ListDir(Tool):
    """List the entries of a directory (directories first, with trailing '/')."""

    name = "list_dir"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "List the immediate entries of a directory in the repository. "
                "Directories are shown first and end with '/'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative directory path. "
                        "Defaults to the repository root.",
                    },
                },
                "required": [],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        rel = args.get("path", ".")
        if not isinstance(rel, str):
            return ToolResult.error("argument 'path' must be a string")
        path = resolve_in_root(ctx.root, rel)
        if not path.exists():
            return ToolResult.error(f"directory not found: {rel}")
        if not path.is_dir():
            return ToolResult.error(f"{rel} is a file, not a directory")

        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return ToolResult.success("(empty directory)")
        listed = "\n".join(
            f"{p.name}/" if p.is_dir() else p.name for p in entries
        )
        return ToolResult.success(truncate(listed, ctx.max_output_chars))


class EditFile(Tool):
    """Exact-string replacement in a file (or create a new file).

    Mirrors the reliable search/replace pattern: the model supplies an exact
    ``old_string`` and its ``new_string``. We refuse ambiguous edits (multiple
    matches) unless ``replace_all`` is set, so the model can't silently change
    the wrong occurrence. An empty ``old_string`` creates a new file.
    """

    name = "edit_file"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Edit a file by replacing an exact substring. 'old_string' must "
                "appear exactly once (unless replace_all=true), preventing "
                "ambiguous edits. To CREATE a new file, pass an empty old_string "
                "and the full contents as new_string."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path to the file.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace. Empty string means "
                        "create a new file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text (or full contents when creating).",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of "
                        "requiring a unique match. Default false.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        )

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        rel = require_str(args, "path")
        old_string = require_str(args, "old_string")
        new_string = require_str(args, "new_string")
        replace_all = bool(args.get("replace_all", False))
        path = resolve_in_root(ctx.root, rel)

        # Creation path: empty old_string.
        if old_string == "":
            if path.exists():
                return ToolResult.error(
                    f"{rel} already exists; pass a non-empty old_string to edit it"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            return ToolResult.success(f"created {rel} ({len(new_string)} chars)")

        # Edit path: file must exist.
        if not path.exists():
            return ToolResult.error(f"file not found: {rel}")
        if path.is_dir():
            return ToolResult.error(f"{rel} is a directory, not a file")

        content = path.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return ToolResult.error(f"old_string not found in {rel}")
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string is ambiguous: found {count} occurrences in {rel}. "
                f"Add more surrounding context to make it unique, or set "
                f"replace_all=true."
            )

        updated = content.replace(old_string, new_string)
        path.write_text(updated, encoding="utf-8")
        n = count if replace_all else 1
        return ToolResult.success(
            f"edited {rel}: replaced {n} occurrence(s)"
        )
