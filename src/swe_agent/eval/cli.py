"""Run the agent on one SWE-bench Lite instance and grade it officially.

Examples:
    python -m swe_agent.eval.cli --instance-id pallets__flask-4045
    python -m swe_agent.eval.cli --instance-id pallets__flask-4045 --max-steps 15

Requires GROQ_API_KEY (or whatever api_key_env config.toml names) and a docker
daemon. The instance image is pulled on first use (multi-GB), then cached.
"""

from __future__ import annotations

import argparse
import sys

from swe_agent.config import load_agent_config, load_config
from swe_agent.dataset import load_swebench_lite
from swe_agent.eval.runner import solve_and_grade
from swe_agent.llm.factory import build_llm_client


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swe-agent-eval", description=__doc__)
    parser.add_argument(
        "--instance-id", required=True, help="SWE-bench Lite instance id to run."
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--max-steps", type=int, help="Override config max_steps.")
    parser.add_argument(
        "--eval-timeout", type=int, default=1800, help="Eval-script timeout (s)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        llm_config = load_config()  # raises if the API key is missing
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    instances = load_swebench_lite(
        split=args.split, instance_ids=[args.instance_id]
    )
    if not instances:
        print(
            f"error: instance {args.instance_id!r} not found in split "
            f"{args.split!r}",
            file=sys.stderr,
        )
        return 2
    instance = instances[0]

    agent_config = load_agent_config()
    if args.max_steps is not None:
        agent_config = type(agent_config)(max_steps=args.max_steps)

    print(f"Instance : {instance.instance_id} ({instance.repo})")
    print(f"Model    : {llm_config.model} (provider: {llm_config.provider})")
    print(f"Max steps: {agent_config.max_steps}")
    print("(first run pulls the instance image, ~GBs)\n")

    outcome = solve_and_grade(
        instance,
        build_llm_client(llm_config),
        agent_config,
        report=print,
        eval_timeout=args.eval_timeout,
    )

    agent_result = outcome.agent_result
    g = outcome.grade
    print("\n" + "=" * 60)
    print(f"stop reason  : {agent_result.stop_reason.value}")
    if agent_result.error:
        print(f"agent error  : {agent_result.error}")
    print(f"steps used   : {agent_result.steps}")
    print(f"patch        : {len(agent_result.patch.splitlines())} diff lines")
    print(f"patch applied: {g.patch_applied}")
    print(f"status       : {g.status}")
    print(
        f"FAIL_TO_PASS : {len(g.fail_to_pass_passed)} passed / "
        f"{len(g.fail_to_pass_failed)} failed"
    )
    print(
        f"PASS_TO_PASS : {len(g.pass_to_pass_passed)} passed / "
        f"{len(g.pass_to_pass_failed)} failed"
    )
    print(f"RESOLVED     : {g.resolved}")
    print("=" * 60)
    return 0 if g.resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
