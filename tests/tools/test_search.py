"""Tests for search_code."""

from __future__ import annotations

from pathlib import Path

from swe_agent.tools.base import ToolContext
from swe_agent.tools.search import SearchCode
from swe_agent.tools.shell import LocalExecutor


def test_literal_match_returns_path_line_text(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "def subtract"})
    assert res.ok
    assert "calculator/ops.py:" in res.output
    assert "def subtract(a, b):" in res.output


def test_no_matches(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "multiply"})
    assert res.ok
    assert "no matches" in res.output


def test_ignores_git_dir_and_binary(tool_ctx: ToolContext) -> None:
    # "subtract" exists in .git/config and data.bin, which must be skipped.
    res = SearchCode().run(tool_ctx, {"pattern": "subtract"})
    assert res.ok
    assert ".git" not in res.output
    assert "data.bin" not in res.output


def test_regex_match(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(
        tool_ctx, {"pattern": r"def (add|subtract)", "regex": True}
    )
    assert res.ok
    assert "def add" in res.output
    assert "def subtract" in res.output


def test_invalid_regex_reported(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "(unclosed", "regex": True})
    assert not res.ok
    assert "invalid regex" in res.output


def test_ignore_case(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "SUBTRACT", "ignore_case": True})
    assert res.ok
    assert "def subtract" in res.output


def test_scoped_to_subpath(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "subtract", "path": "tests"})
    assert res.ok
    assert "tests/test_ops.py" in res.output
    assert "calculator/ops.py" not in res.output


def test_max_results_caps_output(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "a", "max_results": 1})
    assert res.ok
    assert "stopped at max_results=1" in res.output


def test_search_missing_path(tool_ctx: ToolContext) -> None:
    res = SearchCode().run(tool_ctx, {"pattern": "x", "path": "ghost"})
    assert not res.ok
    assert "path not found" in res.output


def test_search_works_when_root_is_a_symlink(dummy_repo: Path, tmp_path: Path) -> None:
    """Regression: a symlinked root (e.g. macOS /var -> /private/var) must not
    break the relative-path computation in results."""
    link = tmp_path / "linked_root"
    link.symlink_to(dummy_repo)
    ctx = ToolContext(root=link, executor=LocalExecutor())
    res = SearchCode().run(ctx, {"pattern": "def subtract"})
    assert res.ok
    assert "calculator/ops.py:" in res.output
