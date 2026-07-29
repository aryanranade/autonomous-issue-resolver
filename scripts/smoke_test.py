"""Cheap end-to-end smoke test: does the whole pipeline actually work?

This exists because a *real* benchmark run is unaffordable on a free tier (a full
25-step attempt costs ~40-60k tokens, and Groq's free tier allows ~100k/day). The
trick here is that most of the pipeline can be validated for **zero LLM tokens**:

    stage            LLM cost   what it proves
    ---------------  ---------  ------------------------------------------------
    grade-gold       none       dataset load, Docker provisioning, x86 emulation,
                                eval-script execution, the 2>&1 marker/stream
                                merge, swebench log parsing, resolution status
    grade-empty      none       the no-diff short circuit (negative control)
    agent            small      the LLM loop: tool calls, editing, patch capture

Only `agent` spends tokens, and it is capped by --max-steps so the spend stays
predictable. It also prints exact token usage so you can budget the next run.

Usage:
    python scripts/smoke_test.py                      # all stages, 4 agent steps
    python scripts/smoke_test.py --stage grade-gold   # free stages only
    python scripts/smoke_test.py --stage agent --max-steps 3

Requires a docker daemon and whatever API key config.toml names (agent stage only).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from swe_agent.config import AgentConfig, load_agent_config, load_config
from swe_agent.dataset import SWEBenchInstance, load_swebench_lite
from swe_agent.eval.grading import grade
from swe_agent.eval.runner import solve_and_grade
from swe_agent.llm.base import LLMClient, LLMResponse, Message, ToolSpec
from swe_agent.llm.factory import build_llm_client

# flask-4045 is the default because its image is small (~3.8GB) relative to the
# django/sympy instances and it is the one already exercised during development.
DEFAULT_INSTANCE = "pallets__flask-4045"

PASS = "PASS"
FAIL = "FAIL"


class CountingLLM(LLMClient):
    """Wraps an LLMClient to accumulate token usage across a whole agent run.

    AgentResult deliberately records *behaviour* (plan, tool calls, transcript)
    rather than cost, so there is no built-in running total. For a smoke test
    whose entire purpose is "will this fit in my daily quota?", the total is the
    headline number — so we count it here rather than changing the core types.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        response = self._inner.complete(messages, tools, **overrides)
        self.calls += 1
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        return response


def _banner(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}", flush=True)


def _verdict(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{PASS if ok else FAIL}] {label}: {detail}", flush=True)
    return ok


def _load(instance_id: str) -> SWEBenchInstance:
    instances = load_swebench_lite(split="test", instance_ids=[instance_id])
    if not instances:
        raise SystemExit(f"error: instance {instance_id!r} not found")
    return instances[0]


def stage_grade_gold(instance: SWEBenchInstance, eval_timeout: int) -> bool:
    """Grade the dataset's own reference patch. Must come back RESOLVED.

    This is the single highest-value free check: if the gold patch does not
    score as resolved, the grading harness is broken and *every* future score is
    meaningless. Costs one container run and zero tokens.
    """
    _banner("STAGE grade-gold (0 tokens) — official grader vs. the gold patch")
    started = time.monotonic()
    result = grade(instance, instance.patch, eval_timeout=eval_timeout)
    elapsed = time.monotonic() - started

    ok = True
    ok &= _verdict("patch applied", result.patch_applied, str(result.patch_applied))
    ok &= _verdict(
        "FAIL_TO_PASS",
        not result.fail_to_pass_failed,
        f"{len(result.fail_to_pass_passed)} passed / "
        f"{len(result.fail_to_pass_failed)} failed",
    )
    ok &= _verdict(
        "PASS_TO_PASS",
        not result.pass_to_pass_failed,
        f"{len(result.pass_to_pass_passed)} passed / "
        f"{len(result.pass_to_pass_failed)} failed",
    )
    ok &= _verdict("resolved", result.resolved, f"{result.resolved} ({result.status})")
    print(f"  ({elapsed:.0f}s)")

    if not ok:
        # Grading failures are usually silent parse problems, so show the tail.
        print("\n  --- last 25 lines of eval output ---")
        for line in result.test_output.splitlines()[-25:]:
            print(f"  | {line}")
    return ok


def stage_grade_empty(instance: SWEBenchInstance) -> bool:
    """An empty diff must be reported as unresolved, not crash. Negative control."""
    _banner("STAGE grade-empty (0 tokens) — no-diff short circuit")
    result = grade(instance, "")
    ok = True
    ok &= _verdict("not resolved", not result.resolved, str(result.resolved))
    ok &= _verdict("status", result.status == "empty_patch", result.status)
    return ok


def stage_agent(
    instance: SWEBenchInstance, agent_config: AgentConfig, eval_timeout: int
) -> bool:
    """Run the real agent loop for a few steps against the real container.

    We do NOT assert that the bug gets resolved — a small open model on a capped
    step budget almost certainly will not, and that is fine. What this proves is
    that the machinery runs: the model is reachable, tools execute inside the
    container, edits land, and a diff is captured and graded without crashing.
    """
    _banner(
        f"STAGE agent (spends tokens) — {agent_config.max_steps} step cap, live LLM"
    )
    llm = CountingLLM(build_llm_client(load_config()))

    started = time.monotonic()
    outcome = solve_and_grade(
        instance, llm, agent_config, report=print, eval_timeout=eval_timeout
    )
    elapsed = time.monotonic() - started
    agent_result = outcome.agent_result

    print()
    ok = True
    ok &= _verdict(
        "llm reachable", llm.calls > 0, f"{llm.calls} completion(s)"
    )
    ok &= _verdict(
        "loop ended cleanly",
        agent_result.error is None,
        agent_result.error or agent_result.stop_reason.value,
    )
    ok &= _verdict(
        "tools executed",
        bool(agent_result.tool_calls),
        f"{len(agent_result.tool_calls)} call(s)",
    )
    ok &= _verdict(
        "grading completed",
        outcome.grade.status != "eval_incomplete",
        outcome.grade.status,
    )
    # Informational only — not a pass/fail criterion at this step budget.
    print(
        f"  [info] produced a diff: {agent_result.made_changes} "
        f"({len(agent_result.patch.splitlines())} lines); "
        f"resolved: {outcome.resolved}"
    )
    print(
        f"\n  tokens: {llm.total_tokens:,} total "
        f"({llm.prompt_tokens:,} in / {llm.completion_tokens:,} out) "
        f"over {llm.calls} call(s)"
    )
    if llm.calls:
        print(
            f"  ~{llm.total_tokens // llm.calls:,} tokens/step -> a full 25-step "
            f"run would cost roughly {(llm.total_tokens // llm.calls) * 25:,}"
        )
    print(f"  ({elapsed:.0f}s)")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke-test", description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "free", "grade-gold", "grade-empty", "agent"],
        help="Which stage to run. 'free' runs only the zero-token stages.",
    )
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Step cap for the agent stage (default: 4, keeps token spend small).",
    )
    parser.add_argument("--eval-timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    instance = _load(args.instance_id)
    print(f"instance: {instance.instance_id} ({instance.repo})")

    stages = {
        "all": ["grade-gold", "grade-empty", "agent"],
        "free": ["grade-gold", "grade-empty"],
    }.get(args.stage, [args.stage])

    results: dict[str, bool] = {}
    for stage in stages:
        if stage == "grade-gold":
            results[stage] = stage_grade_gold(instance, args.eval_timeout)
        elif stage == "grade-empty":
            results[stage] = stage_grade_empty(instance)
        elif stage == "agent":
            config = load_agent_config()
            config = AgentConfig(
                max_steps=args.max_steps,
                keep_recent_tool_results=config.keep_recent_tool_results,
            )
            results[stage] = stage_agent(instance, config, args.eval_timeout)

    _banner("SUMMARY")
    for stage, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {stage}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
