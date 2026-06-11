"""Command-line entry point: run the agent against a repository.

Examples:
    python -m swe_agent.agent.cli --repo ./myrepo --issue "Fix the off-by-one in paginate()"
    python -m swe_agent.agent.cli --repo ./myrepo --issue-file bug.txt --max-steps 15

Requires GROQ_API_KEY (or whatever api_key_env config.toml names) in the
environment. This is the real interface; it streams the agent's tool calls as
they happen and prints the resulting diff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swe_agent.agent.loop import Agent
from swe_agent.config import load_agent_config, load_config
from swe_agent.llm.factory import build_llm_client
from swe_agent.task import Task
from swe_agent.tools.base import ToolContext
from swe_agent.tools.registry import default_registry
from swe_agent.tools.shell import LocalExecutor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swe-agent", description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to the target repository.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--issue", help="Issue text to fix.")
    src.add_argument("--issue-file", type=Path, help="File containing the issue text.")
    parser.add_argument("--task-id", default="cli", help="Identifier for this run.")
    parser.add_argument("--max-steps", type=int, help="Override config max_steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"error: --repo {repo} is not a directory", file=sys.stderr)
        return 2

    problem = args.issue if args.issue else args.issue_file.read_text()

    try:
        llm_config = load_config()  # raises if the API key is missing
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agent_config = load_agent_config()
    if args.max_steps is not None:
        agent_config = type(agent_config)(max_steps=args.max_steps)

    print(f"Model: {llm_config.model} (provider: {llm_config.provider})")
    print(f"Repo:  {repo}")
    print(f"Max steps: {agent_config.max_steps}\n")

    agent = Agent(
        llm=build_llm_client(llm_config),
        registry=default_registry(),
        ctx=ToolContext(root=repo, executor=LocalExecutor()),
        config=agent_config,
    )
    result = agent.run(Task(id=args.task_id, problem_statement=problem), report=print)

    print("\n" + "=" * 60)
    print(f"stop reason : {result.stop_reason.value}")
    print(f"steps used  : {result.steps}")
    print(f"tool calls  : {len(result.tool_calls)}")
    if result.plan:
        print(f"root cause  : {result.plan.root_cause}")
    if result.summary:
        print(f"summary     : {result.summary}")
    print("=" * 60)
    if result.made_changes:
        print("\n--- patch ---\n" + result.patch)
    else:
        print("\n(no changes were made)")

    return 0 if result.finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
