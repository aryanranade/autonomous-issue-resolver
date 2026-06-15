"""Trim the conversation sent to the model, to cut tokens per call.

An agent loop re-sends the whole transcript on every turn, so the bulky
``tool`` outputs (file reads up to 10k chars, full test logs) pile up and
dominate token use — which, on a tokens-per-day-capped free tier, directly
limits how many instances we can grade. The model rarely needs the *verbatim*
text of a file it read many steps ago; it has already acted on it.

So before each call we elide the *content* of all but the most recent few tool
results, replacing them with a short stub. Crucially we do **not** drop any
messages: every assistant ``tool_calls`` turn keeps its matching ``tool``
response (just shortened), preserving the pairing the chat API requires. The
agent's own full transcript is untouched — only the copy sent to the model is
compacted, and the model can always re-read a file if it needs it again.
"""

from __future__ import annotations

from dataclasses import replace

from swe_agent.llm.base import Message

ELIDED = "[earlier tool output elided to save context — re-read the file if you need it]"


def compact_messages(
    messages: list[Message], *, keep_recent_tool_results: int
) -> list[Message]:
    """Return a copy of ``messages`` with old tool-result contents elided.

    The ``keep_recent_tool_results`` most recent ``tool`` messages are kept in
    full; older ones have their ``content`` replaced with a stub. Non-tool
    messages (system, the task, assistant thoughts/tool_calls) are left intact.
    A value <= 0 elides every tool result; a value >= the number of tool
    results leaves the conversation unchanged.
    """
    tool_indices = [i for i, message in enumerate(messages) if message.role == "tool"]
    if keep_recent_tool_results < 0:
        keep_recent_tool_results = 0
    if len(tool_indices) <= keep_recent_tool_results:
        return list(messages)

    keep = tool_indices[-keep_recent_tool_results:] if keep_recent_tool_results else []
    to_elide = set(tool_indices) - set(keep)

    return [
        replace(message, content=ELIDED)
        if index in to_elide and message.content != ELIDED
        else message
        for index, message in enumerate(messages)
    ]
