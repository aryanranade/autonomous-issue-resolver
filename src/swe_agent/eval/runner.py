"""Single-instance runner: solve one SWE-bench instance, then grade it (Phase 3c-ii).

Ties together the agent loop (Phase 2), the real instance environment (3c-i), and
the official grader (3c-ii). Phase 4 will loop this over many instances and
aggregate a success rate; here we run exactly one, end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from swe_agent.agent.loop import Agent, Reporter
from swe_agent.agent.result import AgentResult
from swe_agent.config import AgentConfig
from swe_agent.dataset import SWEBenchInstance
from swe_agent.eval.grading import DEFAULT_EVAL_TIMEOUT, GradeResult, grade
from swe_agent.llm.base import LLMClient
from swe_agent.sandbox.environment import SWEBenchEnvironment
from swe_agent.tools.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)


def _noop(_: str) -> None:
    pass


@dataclass
class InstanceOutcome:
    """The full record of one instance attempt: how the agent did + the score."""

    instance_id: str
    agent_result: AgentResult
    grade: GradeResult

    @property
    def resolved(self) -> bool:
        return self.grade.resolved


def solve_and_grade(
    instance: SWEBenchInstance,
    llm: LLMClient,
    agent_config: AgentConfig,
    *,
    registry: ToolRegistry | None = None,
    report: Reporter = _noop,
    eval_timeout: int = DEFAULT_EVAL_TIMEOUT,
) -> InstanceOutcome:
    """Provision the instance, let the agent attempt it, then grade the patch.

    The agent solves inside a bind-mounted container (its edits + test runs share
    files); grading then runs in a *fresh* container so the score depends only on
    the captured patch, not on any state the agent left behind.
    """
    registry = registry or default_registry()

    with SWEBenchEnvironment(instance) as env:
        agent = Agent(llm, registry, env.tool_context(), agent_config)
        agent_result = agent.run(instance.to_task(), report=report)

    grade_result = grade(instance, agent_result.patch, eval_timeout=eval_timeout)
    logger.info(
        "instance %s: resolved=%s (%s)",
        instance.instance_id,
        grade_result.resolved,
        grade_result.status,
    )
    return InstanceOutcome(instance.instance_id, agent_result, grade_result)
