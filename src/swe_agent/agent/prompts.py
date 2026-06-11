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
4. VERIFY — Use run_tests to confirm the fix works and breaks nothing else. If a
   test fails, read the output and iterate.
5. FINISH — When the relevant tests pass, call finish with a short summary.

Rules:
- Prefer small, targeted edits over rewrites.
- edit_file needs an exact old_string that occurs once in the file — include
  enough surrounding context to make it unique.
- Always investigate before editing and verify after editing.
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
        "Call when the issue is fixed and the relevant tests pass. Ends the run."
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
