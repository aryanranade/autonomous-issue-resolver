"""Run the agent over a batch of SWE-bench Lite instances and score the set.

Examples:
    python -m swe_agent.eval.batch_cli --limit 5
    python -m swe_agent.eval.batch_cli --instance-ids pallets__flask-4045,django__django-11099
    python -m swe_agent.eval.batch_cli --limit 10 --results-dir runs/lite-10

Requires GROQ_API_KEY and docker. Resumable: re-running the same command continues
where an interrupted or rate-limited run left off (already-graded instances are
skipped). On the free tier the daily token cap will usually abort partway — just
re-run tomorrow to pick up the rest.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from swe_agent.config import load_agent_config, load_config
from swe_agent.dataset import SWEBenchInstance, load_swebench_lite
from swe_agent.eval.batch import run_batch
from swe_agent.eval.runner import InstanceOutcome, solve_and_grade
from swe_agent.llm.factory import build_llm_client


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swe-agent-batch", description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--limit", type=int, help="Run the first N instances of the split."
    )
    selection.add_argument(
        "--instance-ids", help="Comma-separated instance ids to run."
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs/lite"),
        help="Where per-instance JSON records are written (default: runs/lite).",
    )
    parser.add_argument("--max-steps", type=int, help="Override config max_steps.")
    parser.add_argument(
        "--eval-timeout", type=int, default=1800, help="Eval-script timeout (s)."
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Don't abort the batch when an instance hits a rate/quota limit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        llm_config = load_config()  # raises if the API key is missing
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ids = (
        [s.strip() for s in args.instance_ids.split(",") if s.strip()]
        if args.instance_ids
        else None
    )
    instances = load_swebench_lite(
        split=args.split, instance_ids=ids, limit=args.limit
    )
    if not instances:
        print("error: no instances selected", file=sys.stderr)
        return 2

    agent_config = load_agent_config()
    if args.max_steps is not None:
        agent_config = replace(agent_config, max_steps=args.max_steps)

    llm = build_llm_client(llm_config)
    print(
        f"Batch: {len(instances)} instance(s)  model={llm_config.model}  "
        f"results={args.results_dir}\n"
    )

    def solve(instance: SWEBenchInstance) -> InstanceOutcome:
        return solve_and_grade(
            instance,
            llm,
            agent_config,
            report=lambda message: print(f"    {instance.instance_id}: {message}"),
            eval_timeout=args.eval_timeout,
        )

    summary = run_batch(
        instances,
        solve,
        results_dir=args.results_dir,
        abort_on_rate_limit=not args.keep_going,
        report=print,
    )

    print("\n" + "=" * 60)
    print(f"instances    : {summary.total}")
    print(f"completed    : {summary.completed}")
    print(f"resolved     : {summary.resolved}")
    print(f"unresolved   : {summary.unresolved}")
    print(f"errored      : {summary.errored}")
    print(f"skipped      : {summary.skipped}")
    print(
        f"resolve rate : {summary.resolve_rate:.1%} "
        f"({summary.resolved}/{summary.total})"
    )
    print(f"results dir  : {summary.results_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
