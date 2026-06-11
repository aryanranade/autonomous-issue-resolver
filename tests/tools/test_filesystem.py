"""Tests for read_file, list_dir, edit_file."""

from __future__ import annotations

from pathlib import Path

from swe_agent.tools.base import ToolContext
from swe_agent.tools.filesystem import EditFile, ListDir, ReadFile
from swe_agent.tools.registry import default_registry

# ----------------------------- read_file ----------------------------------- #


def test_read_file_numbers_lines(tool_ctx: ToolContext) -> None:
    res = ReadFile().run(tool_ctx, {"path": "calculator/ops.py"})
    assert res.ok
    assert "     1\t" in res.output
    assert "def subtract(a, b):" in res.output


def test_read_file_line_range(tool_ctx: ToolContext) -> None:
    res = ReadFile().run(tool_ctx, {"path": "calculator/ops.py", "start_line": 4, "end_line": 5})
    assert res.ok
    lines = res.output.splitlines()
    assert len(lines) == 2
    assert lines[0].lstrip().startswith("4\t")


def test_read_file_missing(tool_ctx: ToolContext) -> None:
    res = ReadFile().run(tool_ctx, {"path": "nope.py"})
    assert not res.ok
    assert "file not found" in res.output


def test_read_file_on_directory(tool_ctx: ToolContext) -> None:
    res = ReadFile().run(tool_ctx, {"path": "calculator"})
    assert not res.ok
    assert "is a directory" in res.output


def test_read_file_path_escape_blocked(tool_ctx: ToolContext) -> None:
    # Path-escape raises ValueError in the tool; the dispatcher (the agent's
    # real entry point) converts it to an error ToolResult.
    res = default_registry().dispatch(
        "read_file", {"path": "../../etc/passwd"}, tool_ctx
    )
    assert not res.ok
    assert "escapes the repository root" in res.output


def test_read_file_inverted_range(tool_ctx: ToolContext) -> None:
    res = ReadFile().run(tool_ctx, {"path": "README.md", "start_line": 5, "end_line": 1})
    assert not res.ok
    assert "empty range" in res.output


# ------------------------------ list_dir ----------------------------------- #


def test_list_dir_root_defaults(tool_ctx: ToolContext) -> None:
    res = ListDir().run(tool_ctx, {})
    assert res.ok
    entries = res.output.splitlines()
    assert "calculator/" in entries
    assert "README.md" in entries
    # directories sort before files
    assert entries.index("calculator/") < entries.index("README.md")


def test_list_dir_subdir(tool_ctx: ToolContext) -> None:
    res = ListDir().run(tool_ctx, {"path": "calculator"})
    assert res.ok
    assert "ops.py" in res.output.splitlines()


def test_list_dir_missing(tool_ctx: ToolContext) -> None:
    res = ListDir().run(tool_ctx, {"path": "ghost"})
    assert not res.ok
    assert "directory not found" in res.output


def test_list_dir_on_file(tool_ctx: ToolContext) -> None:
    res = ListDir().run(tool_ctx, {"path": "README.md"})
    assert not res.ok
    assert "is a file" in res.output


# ------------------------------ edit_file ---------------------------------- #


def test_edit_file_unique_replace(tool_ctx: ToolContext) -> None:
    res = EditFile().run(
        tool_ctx,
        {
            "path": "calculator/ops.py",
            "old_string": "    return a + b  # BUG: should be a - b",
            "new_string": "    return a - b",
        },
    )
    assert res.ok
    content = (tool_ctx.root / "calculator" / "ops.py").read_text()
    assert "return a - b" in content
    assert "BUG" not in content


def test_edit_file_ambiguous_without_replace_all(tool_ctx: ToolContext) -> None:
    # "return a + b" appears in both add() and subtract().
    res = EditFile().run(
        tool_ctx,
        {"path": "calculator/ops.py", "old_string": "return a + b", "new_string": "return 0"},
    )
    assert not res.ok
    assert "ambiguous" in res.output


def test_edit_file_replace_all(tool_ctx: ToolContext) -> None:
    res = EditFile().run(
        tool_ctx,
        {
            "path": "calculator/ops.py",
            "old_string": "return a + b",
            "new_string": "return 0",
            "replace_all": True,
        },
    )
    assert res.ok
    assert "2 occurrence" in res.output


def test_edit_file_old_string_not_found(tool_ctx: ToolContext) -> None:
    res = EditFile().run(
        tool_ctx,
        {"path": "calculator/ops.py", "old_string": "does not exist", "new_string": "x"},
    )
    assert not res.ok
    assert "not found" in res.output


def test_edit_file_creates_new_file(tool_ctx: ToolContext) -> None:
    res = EditFile().run(
        tool_ctx,
        {"path": "pkg/new_module.py", "old_string": "", "new_string": "X = 1\n"},
    )
    assert res.ok
    created = tool_ctx.root / "pkg" / "new_module.py"
    assert created.read_text() == "X = 1\n"


def test_edit_file_create_refuses_existing(tool_ctx: ToolContext) -> None:
    res = EditFile().run(
        tool_ctx, {"path": "README.md", "old_string": "", "new_string": "x"}
    )
    assert not res.ok
    assert "already exists" in res.output


def test_edit_file_path_escape_blocked(tool_ctx: ToolContext) -> None:
    res = default_registry().dispatch(
        "edit_file",
        {"path": "../evil.py", "old_string": "", "new_string": "x"},
        tool_ctx,
    )
    assert not res.ok
    assert "escapes the repository root" in res.output
    assert not (tool_ctx.root.parent / "evil.py").exists()
