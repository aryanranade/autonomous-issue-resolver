"""Tests for the HTML dashboard generator."""

from __future__ import annotations

import json

from swe_agent.eval.dashboard import build_data, render_dashboard


def _rec(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "instance_id": "pallets__flask-4045",
        "repo": "pallets/flask",
        "resolved": True,
        "status": "RESOLVED_FULL",
        "patch_applied": True,
        "error": None,
        "patch": "diff --git a/x b/x\n+added line\n-removed line\n",
        "plan": {"root_cause": "rc", "files_to_change": ["x.py"], "approach": "ap"},
        "tool_calls": [{"step": 1, "name": "edit_file", "ok": True}],
        "fail_to_pass": {"passed": ["t_a"], "failed": []},
        "pass_to_pass": {"passed": ["t_b"], "failed": []},
    }
    base.update(over)
    return base


def test_build_data_classifies_and_summarizes() -> None:
    data = build_data([
        _rec(instance_id="a"),
        _rec(instance_id="b", resolved=False, status="RESOLVED_NO",
             pass_to_pass={"passed": [], "failed": ["t"]}),
    ])
    assert data["summary"]["total"] == 2
    assert data["summary"]["resolved"] == 1
    assert abs(data["summary"]["resolve_rate"] - 0.5) < 1e-9
    outcomes = {o["value"]: o["count"] for o in data["summary"]["outcomes"]}
    assert outcomes["resolved"] == 1
    assert outcomes["regression"] == 1
    assert data["records"][0]["_outcome"] == "resolved"


def test_render_dashboard_is_self_contained_html() -> None:
    html = render_dashboard([_rec()])
    assert html.startswith("<!doctype html>")
    assert "pallets__flask-4045" in html          # data embedded
    assert "/*__DATA__*/" not in html             # placeholder was substituted
    assert "http://" not in html and "https://" not in html  # no external requests
    assert "<script>" in html and "<style>" in html
    # the embedded JSON is valid
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(blob)["summary"]["total"] == 1


def test_render_dashboard_body_only_variant() -> None:
    body = render_dashboard([_rec()], standalone=False)
    assert not body.startswith("<!doctype")
    assert "<style>" in body and 'class="swe"' in body
