"""Tests for SWE-bench Lite row parsing (offline; no dataset download)."""

from __future__ import annotations

from swe_agent.dataset import SWEBenchInstance, instance_from_row

# A row shaped like the hub's, with FAIL_TO_PASS as a JSON string.
ROW = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "abc123",
    "problem_statement": "Modeling's separability_matrix is wrong for nested models",
    "patch": "diff --git a/x.py b/x.py\n",
    "test_patch": "diff --git a/test_x.py b/test_x.py\n",
    "FAIL_TO_PASS": '["test_a", "test_b"]',
    "PASS_TO_PASS": '["test_c"]',
    "environment_setup_commit": "def456",
    "version": "4.3",
}


def test_parses_row_into_instance() -> None:
    inst = instance_from_row(ROW)
    assert isinstance(inst, SWEBenchInstance)
    assert inst.instance_id == "astropy__astropy-12907"
    assert inst.repo == "astropy/astropy"
    assert inst.base_commit == "abc123"
    assert inst.fail_to_pass == ["test_a", "test_b"]
    assert inst.pass_to_pass == ["test_c"]
    assert inst.environment_setup_commit == "def456"
    assert inst.version == "4.3"


def test_fail_to_pass_accepts_real_list() -> None:
    row = {**ROW, "FAIL_TO_PASS": ["t1", "t2"]}
    assert instance_from_row(row).fail_to_pass == ["t1", "t2"]


def test_environment_setup_commit_falls_back_to_base() -> None:
    row = {**ROW}
    del row["environment_setup_commit"]
    assert instance_from_row(row).environment_setup_commit == "abc123"


def test_to_task_extracts_agent_slice() -> None:
    task = instance_from_row(ROW).to_task()
    assert task.id == "astropy__astropy-12907"
    assert "separability_matrix" in task.problem_statement
