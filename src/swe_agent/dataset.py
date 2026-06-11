"""Loading SWE-bench Lite tasks.

SWE-bench Lite is 300 real GitHub issues, each with hidden verifying tests. We
load it from the Hugging Face hub via ``datasets``. Each row becomes a
``SWEBenchInstance`` carrying everything the harness (Phase 3c) needs to build
the environment and grade a fix, plus a ``to_task()`` for the agent loop, which
only needs the id and problem statement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from swe_agent.task import Task

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"


@dataclass
class SWEBenchInstance:
    """One SWE-bench task with the metadata needed to run and grade it."""

    instance_id: str           # e.g. "astropy__astropy-12907"
    repo: str                  # e.g. "astropy/astropy"
    base_commit: str           # commit the repo is checked out at (the buggy state)
    problem_statement: str     # the GitHub issue text the agent must resolve
    patch: str                 # gold (reference) solution patch
    test_patch: str            # test changes that verify the fix
    fail_to_pass: list[str]    # tests that should go from failing -> passing
    pass_to_pass: list[str]    # tests that must remain passing (no regressions)
    environment_setup_commit: str
    version: str

    def to_task(self) -> Task:
        """The slice the agent loop consumes."""
        return Task(id=self.instance_id, problem_statement=self.problem_statement)


def _as_list(value: Any) -> list[str]:
    """Normalize a field that may be a list or a JSON-encoded string list.

    The hub stores FAIL_TO_PASS / PASS_TO_PASS as JSON strings; some mirrors
    decode them to lists. Accept both.
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(v) for v in parsed] if isinstance(parsed, list) else [value]
    return []


def instance_from_row(row: dict[str, Any]) -> SWEBenchInstance:
    """Build a SWEBenchInstance from one dataset row (pure; unit-testable)."""
    return SWEBenchInstance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        patch=row.get("patch", ""),
        test_patch=row.get("test_patch", ""),
        fail_to_pass=_as_list(row.get("FAIL_TO_PASS", [])),
        pass_to_pass=_as_list(row.get("PASS_TO_PASS", [])),
        environment_setup_commit=row.get("environment_setup_commit") or row["base_commit"],
        version=str(row.get("version", "")),
    )


def load_swebench_lite(
    *,
    split: str = "test",
    instance_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[SWEBenchInstance]:
    """Load SWE-bench Lite instances from the Hugging Face hub.

    Args:
        split: "test" (300 instances) or "dev" (23).
        instance_ids: if given, keep only these instance ids.
        limit: cap the number returned (after id filtering) — handy for the
            small subset runs Phase 4 starts with.

    ``datasets`` is imported lazily so importing this module is cheap and only
    pulls the heavy dependency when you actually load the data.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]  # no stubs shipped

    ds = load_dataset(DATASET_NAME, split=split)
    wanted = set(instance_ids) if instance_ids else None

    out: list[SWEBenchInstance] = []
    for row in ds:
        if wanted is not None and row["instance_id"] not in wanted:
            continue
        out.append(instance_from_row(row))
        if limit is not None and len(out) >= limit:
            break
    return out
