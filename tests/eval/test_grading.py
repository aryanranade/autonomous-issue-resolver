"""Tests for official grading (Phase 3c-ii).

The report-translation tests are pure (synthetic swebench reports, no docker).
``test_grade_empty_patch`` short-circuits before any container. The gold-patch
test is the real correctness check — it grades the *reference* solution and must
come back resolved — but it needs docker + the instance image pulled + dataset
access, so it skips otherwise.
"""

from __future__ import annotations

import subprocess

import pytest

from swe_agent.dataset import SWEBenchInstance, load_swebench_lite
from swe_agent.eval.grading import (
    STATUS_EMPTY_PATCH,
    STATUS_EVAL_INCOMPLETE,
    _to_grade_result,
    grade,
)
from swe_agent.sandbox.docker import docker_available

FLASK_ID = "pallets__flask-4045"
FLASK_IMAGE = "swebench/sweb.eval.x86_64.pallets_1776_flask-4045:latest"
FLASK_COMMIT = "d8c37f43724cd9fb0870f77877b7c4c7e38a19e0"


def _report(instance_id: str, *, applied: bool, resolved: bool, tests: dict) -> dict:
    entry = {"patch_successfully_applied": applied, "resolved": resolved}
    if tests:
        entry["tests_status"] = tests
    return {instance_id: entry}


def test_to_grade_result_full_resolution() -> None:
    iid = "x__y-1"
    report = _report(
        iid,
        applied=True,
        resolved=True,
        tests={
            "FAIL_TO_PASS": {"success": ["a"], "failure": []},
            "PASS_TO_PASS": {"success": ["b"], "failure": []},
        },
    )
    result = _to_grade_result(iid, report, "log")
    assert result.resolved is True
    assert result.patch_applied is True
    assert result.status == "RESOLVED_FULL"
    assert result.fail_to_pass_passed == ["a"]
    assert result.pass_to_pass_passed == ["b"]


def test_to_grade_result_partial_is_not_resolved() -> None:
    iid = "x__y-1"
    report = _report(
        iid,
        applied=True,
        resolved=False,
        tests={
            "FAIL_TO_PASS": {"success": ["a"], "failure": ["c"]},
            "PASS_TO_PASS": {"success": ["b"], "failure": []},
        },
    )
    result = _to_grade_result(iid, report, "log")
    assert result.resolved is False
    assert result.status == "RESOLVED_PARTIAL"
    assert result.fail_to_pass_failed == ["c"]


def test_to_grade_result_eval_incomplete() -> None:
    iid = "x__y-1"
    report = _report(iid, applied=False, resolved=False, tests={})
    result = _to_grade_result(iid, report, "log")
    assert result.resolved is False
    assert result.status == STATUS_EVAL_INCOMPLETE


def test_grade_empty_patch_short_circuits_without_docker() -> None:
    instance = SWEBenchInstance(
        instance_id=FLASK_ID,
        repo="pallets/flask",
        base_commit=FLASK_COMMIT,
        problem_statement="x",
        patch="",
        test_patch="",
        fail_to_pass=[],
        pass_to_pass=[],
        environment_setup_commit=FLASK_COMMIT,
        version="2.0",
    )
    result = grade(instance, "   \n  ")
    assert result.resolved is False
    assert result.patch_applied is False
    assert result.status == STATUS_EMPTY_PATCH


def _image_present(ref: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", ref], capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.skipif(not docker_available(), reason="docker daemon not available")
def test_gold_patch_resolves_flask() -> None:
    if not _image_present(FLASK_IMAGE):
        pytest.skip(f"instance image not pulled locally: {FLASK_IMAGE}")
    try:
        instances = load_swebench_lite(instance_ids=[FLASK_ID])
    except Exception as exc:  # noqa: BLE001 — dataset needs network; skip if absent
        pytest.skip(f"could not load dataset: {exc}")
    if not instances:
        pytest.skip(f"{FLASK_ID} not found in dataset")

    instance = instances[0]
    # The reference solution must, by definition, resolve the instance.
    result = grade(instance, instance.patch, eval_timeout=900)
    assert result.patch_applied, result.status
    assert result.resolved, f"{result.status}; f2p_failed={result.fail_to_pass_failed}"
    assert result.fail_to_pass_failed == []
