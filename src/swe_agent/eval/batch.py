"""Batch evaluation harness (Phase 4): run many instances, score the set.

Loops :func:`solve_and_grade` over a list of instances, persists one JSON record
per instance to a results directory, and aggregates a success rate. Designed for
the realities of this project:

* **Resumable** — an instance whose result file already exists is skipped, so an
  interrupted run (or an exhausted token budget) can be continued by re-running.
* **Survivable** — one instance crashing (docker hiccup, grading error) is caught
  and recorded, not allowed to kill the batch.
* **Quota-aware** — if an instance ends because the LLM hit a rate/quota limit,
  continuing is pointless (every remaining instance would do the same), so the
  batch aborts early and leaves the rest for a later resume.

The per-instance work is injected as a ``solve`` callable, which keeps the
orchestration (resume / abort / aggregate / persist) unit-testable without a
real LLM or docker.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_agent.dataset import SWEBenchInstance
from swe_agent.eval.runner import InstanceOutcome

logger = logging.getLogger(__name__)

Reporter = Callable[[str], None]
SolveFn = Callable[[SWEBenchInstance], InstanceOutcome]

# Substrings that mark an error as "the LLM is rate/quota limited" — continuing
# the batch can't help, so we abort and let a later resume pick up the rest.
_RATE_LIMIT_MARKERS = ("ratelimit", "rate_limit", "rate limit", "tokens per", "429")


def _noop(_: str) -> None:
    pass


def _is_rate_limited(error: str | None) -> bool:
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in _RATE_LIMIT_MARKERS)


def outcome_to_record(outcome: InstanceOutcome, repo: str) -> dict[str, Any]:
    """A compact, JSON-safe record of one attempt, tuned for Phase 5 analysis.

    Deliberately omits the full LLM transcript, the raw swebench report, and the
    raw test output — those are large; the patch, plan, tool-call trace, and
    per-test pass/fail lists are what failure analysis actually needs.
    """
    agent = outcome.agent_result
    grade = outcome.grade
    plan = agent.plan
    return {
        "instance_id": outcome.instance_id,
        "repo": repo,
        "resolved": grade.resolved,
        "status": grade.status,
        "patch_applied": grade.patch_applied,
        "stop_reason": agent.stop_reason.value,
        "steps": agent.steps,
        "error": agent.error,
        "summary": agent.summary,
        "plan": (
            None
            if plan is None
            else {
                "root_cause": plan.root_cause,
                "files_to_change": plan.files_to_change,
                "approach": plan.approach,
            }
        ),
        "tool_calls": [
            {"step": r.step, "name": r.name, "ok": r.ok} for r in agent.tool_calls
        ],
        "fail_to_pass": {
            "passed": grade.fail_to_pass_passed,
            "failed": grade.fail_to_pass_failed,
        },
        "pass_to_pass": {
            "passed": grade.pass_to_pass_passed,
            "failed": grade.pass_to_pass_failed,
        },
        "patch": agent.patch,
    }


def _error_record(instance: SWEBenchInstance, exc: Exception) -> dict[str, Any]:
    """Record for an instance whose solve/grade raised (infra failure)."""
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "resolved": False,
        "status": "run_error",
        "patch_applied": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


@dataclass
class BatchSummary:
    """Aggregate outcome of a batch over a fixed set of instances."""

    total: int          # instances in scope
    completed: int      # have a result file (run this time or a prior resume)
    resolved: int       # graded as resolved
    unresolved: int     # attempted, graded, not resolved
    errored: int        # solve/grade failed or the agent ended on an error
    skipped: int        # in scope but never run (e.g. aborted after a rate limit)
    results_dir: str

    @property
    def resolve_rate(self) -> float:
        """Resolved over the full scope — the honest benchmark number."""
        return self.resolved / self.total if self.total else 0.0


def summarize_batch(
    instances: list[SWEBenchInstance], results_dir: Path
) -> BatchSummary:
    """Aggregate a summary by reading the result files for ``instances``."""
    resolved = unresolved = errored = completed = 0
    for instance in instances:
        path = results_dir / f"{instance.instance_id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        completed += 1
        if record.get("status") == "run_error" or record.get("error"):
            errored += 1
        elif record.get("resolved"):
            resolved += 1
        else:
            unresolved += 1
    total = len(instances)
    return BatchSummary(
        total=total,
        completed=completed,
        resolved=resolved,
        unresolved=unresolved,
        errored=errored,
        skipped=total - completed,
        results_dir=str(results_dir),
    )


def run_batch(
    instances: list[SWEBenchInstance],
    solve: SolveFn,
    *,
    results_dir: Path,
    abort_on_rate_limit: bool = True,
    report: Reporter = _noop,
) -> BatchSummary:
    """Run ``solve`` over ``instances``, persisting a record per instance.

    Args:
        instances: the instances in scope (the benchmark subset).
        solve: per-instance work, e.g. a closure over ``solve_and_grade``.
        results_dir: where ``<instance_id>.json`` records are written/read.
        abort_on_rate_limit: stop early if an instance hits an LLM rate/quota
            limit (the rest would too); already-written records let you resume.
        report: progress sink (the CLI passes ``print``).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    total = len(instances)
    aborted = False

    for index, instance in enumerate(instances, start=1):
        path = results_dir / f"{instance.instance_id}.json"
        prefix = f"[{index}/{total}] {instance.instance_id}"

        if path.exists():
            report(f"{prefix}: already done, skipping")
            continue
        if aborted:
            continue

        report(f"{prefix}: running")
        try:
            outcome = solve(instance)
            record = outcome_to_record(outcome, instance.repo)
            rate_limited = _is_rate_limited(outcome.agent_result.error)
        except Exception as exc:  # noqa: BLE001 — keep the batch alive
            logger.exception("instance %s crashed", instance.instance_id)
            record = _error_record(instance, exc)
            rate_limited = _is_rate_limited(record["error"])

        _write_json(path, record)
        report(f"{prefix}: {record['status']} (resolved={record['resolved']})")

        if abort_on_rate_limit and rate_limited:
            aborted = True
            report(
                f"rate/quota limit hit; aborting the remaining "
                f"{total - index} instance(s) — resume later to continue"
            )

    return summarize_batch(instances, results_dir)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
