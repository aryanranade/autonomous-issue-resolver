"""Tests for the tool registry / dispatcher."""

from __future__ import annotations

from swe_agent.tools.base import ToolContext
from swe_agent.tools.registry import default_registry


def test_default_registry_exposes_all_tools() -> None:
    reg = default_registry()
    assert set(reg.names()) == {
        "read_file",
        "list_dir",
        "search_code",
        "edit_file",
        "run_shell",
        "run_tests",
    }


def test_specs_match_tool_names() -> None:
    reg = default_registry()
    spec_names = {spec.name for spec in reg.specs()}
    assert spec_names == set(reg.names())


def test_dispatch_runs_named_tool(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    res = reg.dispatch("read_file", {"path": "README.md"}, tool_ctx)
    assert res.ok
    assert "Dummy" in res.output


def test_dispatch_unknown_tool(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    res = reg.dispatch("frobnicate", {}, tool_ctx)
    assert not res.ok
    assert "unknown tool" in res.output


def test_dispatch_converts_missing_arg_to_error(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    res = reg.dispatch("read_file", {}, tool_ctx)  # missing required 'path'
    assert not res.ok
    assert "missing required argument 'path'" in res.output


def test_dispatch_never_raises_on_bad_type(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    # path should be a string; passing an int must become an error, not a crash.
    res = reg.dispatch("read_file", {"path": 123}, tool_ctx)
    assert not res.ok


def test_duplicate_tool_names_rejected() -> None:
    from swe_agent.tools.filesystem import ReadFile
    from swe_agent.tools.registry import ToolRegistry

    try:
        ToolRegistry([ReadFile(), ReadFile()])
    except ValueError as exc:
        assert "duplicate tool name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for duplicate tool names")
