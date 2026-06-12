"""Failure analysis (Phase 5): turn run records into a success rate + breakdown.

Reads the per-instance JSON records written by the batch harness (Phase 4) and
classifies each into a single, prioritised outcome — resolved or one of several
failure modes — then aggregates the citable numbers that are this project's
deliverable: the resolve rate and *why* the rest failed.

The classification is the analytical core, so it's a pure function over the
record dict and is unit-tested exhaustively. The taxonomy is intentionally small
and ordered by primary cause (the first matching rule wins), so every instance
maps to exactly one informative bucket.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Outcome(str, Enum):
    """Single primary outcome for an instance. str-valued for clean JSON/printing."""

    RESOLVED = "resolved"                  # fixed it: all FAIL_TO_PASS pass, no regressions
    REGRESSION = "regression"              # patch broke previously-passing tests
    INCOMPLETE_FIX = "incomplete_fix"      # no regressions, but FAIL_TO_PASS not all passing
    NO_PATCH = "no_patch"                  # agent produced no diff
    PATCH_FAILED = "patch_apply_failed"    # patch didn't apply to a clean checkout
    EVAL_INCOMPLETE = "eval_incomplete"    # eval couldn't be parsed
    LLM_ERROR = "llm_error"                # run ended on an LLM failure (e.g. quota)
    RUN_ERROR = "run_error"                # provisioning/grading crashed


# Display order for the breakdown table (success first, then by how "early" the
# failure is in the pipeline).
OUTCOME_ORDER = [
    Outcome.RESOLVED,
    Outcome.REGRESSION,
    Outcome.INCOMPLETE_FIX,
    Outcome.NO_PATCH,
    Outcome.PATCH_FAILED,
    Outcome.EVAL_INCOMPLETE,
    Outcome.LLM_ERROR,
    Outcome.RUN_ERROR,
]


def classify(record: dict[str, Any]) -> Outcome:
    """Map one result record to its primary outcome (first matching rule wins)."""
    if record.get("resolved"):
        return Outcome.RESOLVED
    if record.get("status") == "run_error":
        return Outcome.RUN_ERROR
    if record.get("error"):
        # The agent ended on a terminal LLM failure (quota/network) — the run was
        # cut short, so this is the primary cause regardless of any partial patch.
        return Outcome.LLM_ERROR
    if record.get("status") == "apply_failed":
        return Outcome.PATCH_FAILED
    if record.get("status") == "empty_patch" or not str(record.get("patch") or "").strip():
        return Outcome.NO_PATCH
    if record.get("status") == "eval_incomplete":
        return Outcome.EVAL_INCOMPLETE
    pass_to_pass = record.get("pass_to_pass") or {}
    if pass_to_pass.get("failed"):
        return Outcome.REGRESSION
    return Outcome.INCOMPLETE_FIX


@dataclass
class AnalysisReport:
    """Aggregate analysis over a set of result records."""

    total: int
    resolved: int
    outcome_counts: dict[str, int]            # outcome value -> count
    per_repo: dict[str, list[int]]            # repo -> [total, resolved]
    per_instance: list[tuple[str, str]] = field(default_factory=list)  # (id, outcome)

    @property
    def resolve_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


def analyze(records: list[dict[str, Any]]) -> AnalysisReport:
    """Classify every record and aggregate the counts."""
    outcome_counts = {outcome.value: 0 for outcome in Outcome}
    per_repo: dict[str, list[int]] = {}
    per_instance: list[tuple[str, str]] = []
    resolved = 0

    for record in records:
        outcome = classify(record)
        outcome_counts[outcome.value] += 1
        instance_id = str(record.get("instance_id", "?"))
        per_instance.append((instance_id, outcome.value))

        repo = str(record.get("repo", "?"))
        slot = per_repo.setdefault(repo, [0, 0])
        slot[0] += 1
        if outcome is Outcome.RESOLVED:
            slot[1] += 1
            resolved += 1

    return AnalysisReport(
        total=len(records),
        resolved=resolved,
        outcome_counts=outcome_counts,
        per_repo=per_repo,
        per_instance=per_instance,
    )


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` result record in ``results_dir`` (sorted by id)."""
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "report.json":
            continue
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("skipping unparseable record: %s", path)
    return records


def format_report(report: AnalysisReport) -> str:
    """Render a Markdown report — the citable deliverable."""
    lines = [
        "# SWE-bench Lite — Agent Evaluation Report",
        "",
        f"- Instances graded: **{report.total}**",
        f"- Resolved: **{report.resolved}** "
        f"(**{report.resolve_rate:.1%}** resolve rate)",
        "",
        "## Outcome breakdown",
        "",
        "| outcome | count | share |",
        "|---|---:|---:|",
    ]
    for outcome in OUTCOME_ORDER:
        count = report.outcome_counts.get(outcome.value, 0)
        if count == 0:
            continue
        share = count / report.total if report.total else 0.0
        lines.append(f"| {outcome.value} | {count} | {share:.0%} |")

    if len(report.per_repo) > 1:
        lines += ["", "## By repository", "", "| repo | instances | resolved | rate |",
                  "|---|---:|---:|---:|"]
        for repo in sorted(report.per_repo):
            total, resolved = report.per_repo[repo]
            rate = resolved / total if total else 0.0
            lines.append(f"| {repo} | {total} | {resolved} | {rate:.0%} |")

    lines += ["", "## Per-instance", "", "| instance | outcome |", "|---|---|"]
    for instance_id, outcome_value in report.per_instance:
        lines.append(f"| {instance_id} | {outcome_value} |")

    return "\n".join(lines) + "\n"
