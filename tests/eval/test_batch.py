"""Tests for the batch harness (Phase 4).

The per-instance work is injected as a fake ``solve``, so resume / abort /
aggregation / persistence are exercised with no LLM and no docker.
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_agent.agent.result import AgentResult, Plan, StopReason, ToolCallRecord
from swe_agent.dataset import SWEBenchInstance
from swe_agent.eval.batch import outcome_to_record, run_batch
from swe_agent.eval.grading import GradeResult
from swe_agent.eval.runner import InstanceOutcome


def _inst(instance_id: str) -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id=instance_id,
        repo="acme/widget",
        base_commit="c0ffee",
        problem_statement="x",
        patch="",
        test_patch="",
        fail_to_pass=["t_a"],
        pass_to_pass=["t_b"],
        environment_setup_commit="c0ffee",
        version="1.0",
    )


def _outcome(
    instance_id: str,
    *,
    resolved: bool,
    status: str = "RESOLVED_NO",
    error: str | None = None,
    patch: str = "diff --git a/x b/x\n",
) -> InstanceOutcome:
    agent = AgentResult(
        task_id=instance_id,
        stop_reason=StopReason.ERROR if error else StopReason.FINISHED,
        finished=error is None,
        steps=3,
        plan=Plan(root_cause="rc", files_to_change=["x.py"], approach="ap"),
        summary="did a thing",
        patch=patch,
        tool_calls=[ToolCallRecord(step=1, name="edit_file", arguments={}, ok=True,
                                   output_preview="ok")],
        error=error,
    )
    grade = GradeResult(
        instance_id=instance_id,
        resolved=resolved,
        patch_applied=True,
        status=status,
        fail_to_pass_passed=["t_a"] if resolved else [],
        fail_to_pass_failed=[] if resolved else ["t_a"],
        pass_to_pass_passed=["t_b"],
    )
    return InstanceOutcome(instance_id, agent, grade)


def test_outcome_to_record_is_json_safe_and_compact() -> None:
    record = outcome_to_record(_outcome("acme__widget-1", resolved=True,
                                        status="RESOLVED_FULL"), "acme/widget")
    # round-trips through JSON
    reloaded = json.loads(json.dumps(record))
    assert reloaded["instance_id"] == "acme__widget-1"
    assert reloaded["resolved"] is True
    assert reloaded["status"] == "RESOLVED_FULL"
    assert reloaded["plan"]["files_to_change"] == ["x.py"]
    assert reloaded["tool_calls"] == [{"step": 1, "name": "edit_file", "ok": True}]
    assert reloaded["fail_to_pass"]["passed"] == ["t_a"]
    # heavy fields are intentionally excluded
    assert "transcript" not in reloaded
    assert "test_output" not in reloaded


def test_run_batch_writes_records_and_aggregates(tmp_path: Path) -> None:
    instances = [_inst("a-1"), _inst("b-2"), _inst("c-3")]
    table = {
        "a-1": _outcome("a-1", resolved=True, status="RESOLVED_FULL"),
        "b-2": _outcome("b-2", resolved=False),
        "c-3": _outcome("c-3", resolved=False, error="boom"),
    }
    summary = run_batch(
        instances, lambda i: table[i.instance_id], results_dir=tmp_path
    )

    assert summary.total == 3
    assert summary.completed == 3
    assert summary.resolved == 1
    assert summary.unresolved == 1
    assert summary.errored == 1  # the one with an agent error
    assert summary.skipped == 0
    assert abs(summary.resolve_rate - 1 / 3) < 1e-9
    # a record file exists for each
    assert {p.stem for p in tmp_path.glob("*.json")} == {"a-1", "b-2", "c-3"}


def test_run_batch_resumes_skipping_existing(tmp_path: Path) -> None:
    # Pre-seed a result for a-1; solve must not be called for it.
    (tmp_path / "a-1.json").write_text(
        json.dumps({"instance_id": "a-1", "resolved": True, "status": "RESOLVED_FULL"})
    )
    called: list[str] = []

    def solve(instance: SWEBenchInstance) -> InstanceOutcome:
        called.append(instance.instance_id)
        return _outcome(instance.instance_id, resolved=False)

    summary = run_batch([_inst("a-1"), _inst("b-2")], solve, results_dir=tmp_path)

    assert called == ["b-2"]  # a-1 was skipped (already done)
    assert summary.completed == 2
    assert summary.resolved == 1  # the pre-seeded a-1 counts


def test_run_batch_aborts_on_rate_limit(tmp_path: Path) -> None:
    instances = [_inst("a-1"), _inst("b-2"), _inst("c-3")]
    table = {
        "a-1": _outcome("a-1", resolved=False),
        "b-2": _outcome("b-2", resolved=False, status="empty_patch",
                        error="RateLimitError: tokens per day exhausted"),
        # c-3 should never be attempted
    }
    called: list[str] = []

    def solve(instance: SWEBenchInstance) -> InstanceOutcome:
        called.append(instance.instance_id)
        return table[instance.instance_id]

    summary = run_batch(instances, solve, results_dir=tmp_path)

    assert called == ["a-1", "b-2"]  # aborted before c-3
    assert summary.completed == 2
    assert summary.skipped == 1
    assert not (tmp_path / "c-3.json").exists()


def test_run_batch_survives_a_crashing_instance(tmp_path: Path) -> None:
    def solve(instance: SWEBenchInstance) -> InstanceOutcome:
        if instance.instance_id == "b-2":
            raise RuntimeError("docker exploded")
        return _outcome(instance.instance_id, resolved=True, status="RESOLVED_FULL")

    summary = run_batch(
        [_inst("a-1"), _inst("b-2"), _inst("c-3")], solve, results_dir=tmp_path
    )

    assert summary.completed == 3  # all three have records
    assert summary.resolved == 2
    assert summary.errored == 1
    crashed = json.loads((tmp_path / "b-2.json").read_text())
    assert crashed["status"] == "run_error"
    assert "docker exploded" in crashed["error"]
