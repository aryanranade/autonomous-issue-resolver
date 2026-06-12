"""System prompt and control-tool specs for the agent loop.

Kept in one place so the prompt can be tuned independently of the loop logic
(prompt quality is the single biggest lever on solve rate for a weak model).
"""

from __future__ import annotations

from swe_agent.llm.base import Message, ToolSpec
from swe_agent.task import Task

SYSTEM_PROMPT = """\
You are an autonomous software engineer. You are given an issue (a bug report or
feature request) for a Python project and must fix it by editing the repository.

You act only by calling tools. Follow this workflow:

1. INVESTIGATE — Use search_code, list_dir, and read_file to find the code
   relevant to the issue and understand the root cause. Do not guess; read the
   actual code first.
2. PLAN — Once you understand the problem, call record_plan with the root cause,
   the file(s) you will change, and your approach. Do this before any edit.
3. FIX — Use edit_file to make the smallest change that resolves the issue.
   Keep existing behavior and style intact everywhere else.
4. VERIFY — After editing, use run_tests to check for regressions, not just your
   one case. Run the WHOLE test file(s) covering the code you changed (e.g. if
   you edited foo/bar.py, run the entire tests/test_bar.py), and ideally the
   surrounding test directory. A change that makes a previously-passing test
   fail is a broken fix — read the failure and narrow or rethink your edit.
5. FINISH — Only call finish once run_tests shows BOTH that the issue is fixed
   AND that no test which passed before your change now fails.

Rules:
- Prefer small, targeted edits over rewrites. The narrowest change that resolves
  the issue is almost always the right one.
- Breaking existing, previously-passing tests is a FAILED fix — graders count
  regressions against you. When in doubt, scope your change more tightly.
- edit_file needs an exact old_string that occurs once in the file — include
  enough surrounding context to make it unique.
- Always investigate before editing, and run the broad tests after editing.
- If an approach fails, rethink it rather than repeating the same action.
- Your step budget is limited; be efficient.
"""


def build_task_messages(task: Task) -> list[Message]:
    """Initial system + user messages for a task."""
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Issue to fix:\n\n{task.problem_statement}\n\n"
                "Begin by investigating the repository to locate the relevant code."
            ),
        ),
    ]


# --- Control tools: intercepted by the loop, never dispatched to the registry --

RECORD_PLAN_SPEC = ToolSpec(
    name="record_plan",
    description=(
        "Record your diagnosis and plan once you understand the root cause and "
        "before making any edit. Call this exactly once."
    ),
    parameters={
        "type": "object",
        "properties": {
            "root_cause": {
                "type": "string",
                "description": "What is actually causing the issue.",
            },
            "files_to_change": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repository-relative paths you intend to edit.",
            },
            "approach": {
                "type": "string",
                "description": "How you will fix it, in one or two sentences.",
            },
        },
        "required": ["root_cause", "files_to_change", "approach"],
    },
)

FINISH_SPEC = ToolSpec(
    name="finish",
    description=(
        "Call only after run_tests confirms the issue is fixed AND no previously "
        "passing test now fails. Ends the run."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Short summary of what you changed and why.",
            },
        },
        "required": ["summary"],
    },
)
