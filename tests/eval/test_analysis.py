"""Tests for failure analysis (Phase 5) — pure functions over record dicts."""

from __future__ import annotations

import json
from pathlib import Path

from swe_agent.eval.analysis import (
    Outcome,
    analyze,
    classify,
    format_report,
    load_records,
)


def _rec(**over: object) -> dict[str, object]:
    """A resolved record by default; override fields to make failure cases."""
    base: dict[str, object] = {
        "instance_id": "acme__widget-1",
        "repo": "acme/widget",
        "resolved": True,
        "status": "RESOLVED_FULL",
        "patch_applied": True,
        "error": None,
        "patch": "diff --git a/x b/x\n",
        "fail_to_pass": {"passed": ["t_a"], "failed": []},
        "pass_to_pass": {"passed": ["t_b"], "failed": []},
    }
    base.update(over)
    return base


def test_classify_resolved() -> None:
    assert classify(_rec()) is Outcome.RESOLVED


def test_classify_regression() -> None:
    rec = _rec(resolved=False, status="RESOLVED_NO",
               fail_to_pass={"passed": ["t_a"], "failed": []},
               pass_to_pass={"passed": [], "failed": ["t_b"]})
    assert classify(rec) is Outcome.REGRESSION


def test_classify_incomplete_fix() -> None:
    # patch applied, no regressions, but the target test still fails
    rec = _rec(resolved=False, status="RESOLVED_NO",
               fail_to_pass={"passed": [], "failed": ["t_a"]},
               pass_to_pass={"passed": ["t_b"], "failed": []})
    assert classify(rec) is Outcome.INCOMPLETE_FIX


def test_classify_no_patch_from_status() -> None:
    rec = _rec(resolved=False, status="empty_patch", patch="", patch_applied=False)
    assert classify(rec) is Outcome.NO_PATCH


def test_classify_no_patch_from_empty_diff() -> None:
    rec = _rec(resolved=False, status="RESOLVED_NO", patch="   \n")
    assert classify(rec) is Outcome.NO_PATCH


def test_classify_patch_failed() -> None:
    rec = _rec(resolved=False, status="apply_failed", patch_applied=False,
               patch="diff --git a/x b/x\n")
    assert classify(rec) is Outcome.PATCH_FAILED


def test_classify_llm_error_takes_priority_over_no_patch() -> None:
    rec = _rec(resolved=False, status="empty_patch", patch="",
               error="RateLimitError: tokens per day exhausted")
    assert classify(rec) is Outcome.LLM_ERROR


def test_classify_run_error() -> None:
    rec = {"instance_id": "x", "repo": "r", "resolved": False,
           "status": "run_error", "error": "DockerError: boom"}
    assert classify(rec) is Outcome.RUN_ERROR


def test_analyze_aggregates_counts_and_per_repo() -> None:
    records = [
        _rec(instance_id="a", repo="acme/widget"),  # resolved
        _rec(instance_id="b", repo="acme/widget", resolved=False, status="RESOLVED_NO",
             pass_to_pass={"passed": [], "failed": ["t"]}),  # regression
        _rec(instance_id="c", repo="other/lib", resolved=False, status="empty_patch",
             patch=""),  # no_patch
    ]
    report = analyze(records)

    assert report.total == 3
    assert report.resolved == 1
    assert abs(report.resolve_rate - 1 / 3) < 1e-9
    assert report.outcome_counts[Outcome.RESOLVED.value] == 1
    assert report.outcome_counts[Outcome.REGRESSION.value] == 1
    assert report.outcome_counts[Outcome.NO_PATCH.value] == 1
    assert report.per_repo["acme/widget"] == [2, 1]
    assert report.per_repo["other/lib"] == [1, 0]


def test_format_report_contains_headline_numbers() -> None:
    report = analyze([_rec(), _rec(instance_id="b", resolved=False,
                                   status="RESOLVED_NO",
                                   pass_to_pass={"passed": [], "failed": ["t"]})])
    text = format_report(report)
    assert "# SWE-bench Lite — Agent Evaluation Report" in text
    assert "50.0%" in text  # 1 of 2 resolved
    assert "| regression |" in text
    assert "acme__widget-1" in text


def test_load_records_reads_dir_and_skips_report(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_rec(instance_id="a")))
    (tmp_path / "b.json").write_text(json.dumps(_rec(instance_id="b")))
    (tmp_path / "report.json").write_text(json.dumps({"not": "a record"}))

    records = load_records(tmp_path)
    ids = sorted(r["instance_id"] for r in records)
    assert ids == ["a", "b"]  # report.json excluded
