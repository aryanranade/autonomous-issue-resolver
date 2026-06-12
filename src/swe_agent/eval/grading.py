"""Official SWE-bench grading: did the agent's patch resolve the instance?

We do **not** reimplement scoring. We mirror ``swebench.harness.run_evaluation``
.``run_instance`` — apply the model patch, run the instance's ``eval_script``,
parse the log with swebench's grader — but drive it through our own
``DockerSandbox`` instead of swebench's docker-SDK orchestration (which assumes
x86_64 build infrastructure). The pass/fail produced here is the same one the
SWE-bench leaderboard uses.

Faithful-replication details that matter:

* The patch is applied in a **fresh** container from the instance image, not the
  agent's mutated environment, so grading depends only on (instance, patch).
* swebench tries three apply commands in order (``GIT_APPLY_CMDS``); we do too.
* ``eval_script`` brackets the test output with ``>>>>> Start/End Test Output``,
  but those lines come from bash's ``set -x`` trace (**stderr**) while pytest
  prints to **stdout**. We run the script with ``2>&1`` so the two streams merge
  in chronological order — otherwise the log parser can't split on the markers.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swebench.harness.constants import (  # type: ignore[import-untyped]
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    PASS_TO_PASS,
)
from swebench.harness.grading import (  # type: ignore[import-untyped]
    get_eval_report,
    get_resolution_status,
)

from swe_agent.dataset import SWEBenchInstance
from swe_agent.sandbox.docker import DockerSandbox
from swe_agent.sandbox.environment import PLATFORM, instance_image_ref, make_spec

logger = logging.getLogger(__name__)

# The exact apply attempts swebench makes, in order (run_evaluation.py).
GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)
CONTAINER_PATCH = "/tmp/patch.diff"  # swebench's DOCKER_PATCH
CONTAINER_EVAL = "/eval.sh"
DEFAULT_EVAL_TIMEOUT = 1800  # seconds; emulated runs are slow, so be generous

# Statuses for cases that never reach the official grader.
STATUS_EMPTY_PATCH = "empty_patch"
STATUS_APPLY_FAILED = "apply_failed"
STATUS_EVAL_INCOMPLETE = "eval_incomplete"  # markers missing / tests errored


@dataclass
class GradeResult:
    """Outcome of grading one patch against one instance."""

    instance_id: str
    resolved: bool          # the leaderboard criterion: all F2P pass AND all P2P pass
    patch_applied: bool     # did the model patch apply to the repo at all?
    status: str             # one of STATUS_* above, or swebench's RESOLVED_FULL/PARTIAL/NO
    fail_to_pass_passed: list[str] = field(default_factory=list)
    fail_to_pass_failed: list[str] = field(default_factory=list)
    pass_to_pass_passed: list[str] = field(default_factory=list)
    pass_to_pass_failed: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)  # raw swebench report
    test_output: str = ""   # full eval-script output (for failure analysis)


def grade(
    instance: SWEBenchInstance,
    patch: str,
    *,
    platform: str = PLATFORM,
    eval_timeout: int = DEFAULT_EVAL_TIMEOUT,
    model_name: str = "swe-agent",
) -> GradeResult:
    """Apply ``patch`` to a fresh instance container and grade it officially."""
    if not patch.strip():
        # No diff -> the failing tests stay failing; don't waste a container.
        return GradeResult(
            instance.instance_id,
            resolved=False,
            patch_applied=False,
            status=STATUS_EMPTY_PATCH,
        )

    spec = make_spec(instance)
    image = instance_image_ref(instance)

    with DockerSandbox(image, platform=platform) as sandbox:
        if not _apply_patch(sandbox, patch):
            return GradeResult(
                instance.instance_id,
                resolved=False,
                patch_applied=False,
                status=STATUS_APPLY_FAILED,
            )
        test_output = _run_eval_script(sandbox, str(spec.eval_script), eval_timeout)

    report_map = _parse_report(spec, instance.instance_id, patch, model_name, test_output)
    return _to_grade_result(instance.instance_id, report_map, test_output)


def _apply_patch(sandbox: DockerSandbox, patch: str) -> bool:
    """Drop the patch into the container and try swebench's apply commands."""
    _write_container_file(sandbox, CONTAINER_PATCH, patch)
    for cmd in GIT_APPLY_CMDS:
        result = sandbox.exec(f"{cmd} {CONTAINER_PATCH}")
        if result.exit_code == 0:
            logger.info("applied patch with %r", cmd)
            return True
    logger.info("patch failed to apply with all of %s", list(GIT_APPLY_CMDS))
    return False


def _run_eval_script(sandbox: DockerSandbox, eval_script: str, timeout: int) -> str:
    """Run the official eval script, merging stderr into stdout for the markers."""
    _write_container_file(sandbox, CONTAINER_EVAL, eval_script)
    result = sandbox.exec(f"/bin/bash {CONTAINER_EVAL} 2>&1", timeout=timeout)
    return result.stdout + result.stderr


def _parse_report(
    spec: Any,
    instance_id: str,
    patch: str,
    model_name: str,
    test_output: str,
) -> dict[str, Any]:
    """Run swebench's grader over the eval log (it reads a file on disk)."""
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: model_name,
        KEY_PREDICTION: patch,
    }
    log_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(test_output)
            log_path = handle.name
        report: dict[str, Any] = get_eval_report(
            spec, prediction, log_path, include_tests_status=True
        )
        return report
    finally:
        if log_path:
            Path(log_path).unlink(missing_ok=True)


def _to_grade_result(
    instance_id: str, report_map: dict[str, Any], test_output: str
) -> GradeResult:
    """Translate swebench's nested report into a flat GradeResult."""
    info: dict[str, Any] = report_map.get(instance_id, {})
    tests: dict[str, Any] = info.get("tests_status") or {}
    f2p: dict[str, list[str]] = tests.get(FAIL_TO_PASS, {})
    p2p: dict[str, list[str]] = tests.get(PASS_TO_PASS, {})
    resolved = bool(info.get("resolved", False))

    if not info.get("patch_successfully_applied", False):
        # The model patch applied, but the eval log was unusable (no markers /
        # tests errored), so swebench couldn't score it.
        status = STATUS_EVAL_INCOMPLETE
    else:
        status = str(get_resolution_status(tests))

    return GradeResult(
        instance_id=instance_id,
        resolved=resolved,
        patch_applied=True,
        status=status,
        fail_to_pass_passed=list(f2p.get("success", [])),
        fail_to_pass_failed=list(f2p.get("failure", [])),
        pass_to_pass_passed=list(p2p.get("success", [])),
        pass_to_pass_failed=list(p2p.get("failure", [])),
        report=report_map,
        test_output=test_output,
    )


def _write_container_file(sandbox: DockerSandbox, container_path: str, content: str) -> None:
    """Write ``content`` to ``container_path`` via a host temp file + docker cp."""
    host_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(content)
            host_path = handle.name
        sandbox.copy_in(host_path, container_path)
    finally:
        if host_path:
            Path(host_path).unlink(missing_ok=True)
