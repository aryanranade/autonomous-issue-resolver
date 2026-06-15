"""Tests for context compaction (token-saving message trimming)."""

from __future__ import annotations

from swe_agent.agent.compaction import ELIDED, compact_messages
from swe_agent.llm.base import Message, ToolCall


def _conversation(num_tool_results: int) -> list[Message]:
    """system + task, then N (assistant tool_call, tool result) pairs."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
    ]
    for i in range(num_tool_results):
        call_id = f"c{i}"
        msgs.append(
            Message(
                role="assistant",
                tool_calls=[ToolCall(id=call_id, name="read_file", arguments={})],
            )
        )
        msgs.append(
            Message(
                role="tool",
                content=f"big file contents {i}" * 100,
                tool_call_id=call_id,
                name="read_file",
            )
        )
    return msgs


def test_no_trimming_when_under_limit() -> None:
    msgs = _conversation(3)
    out = compact_messages(msgs, keep_recent_tool_results=6)
    assert out == msgs  # unchanged


def test_elides_old_tool_outputs_keeps_recent() -> None:
    msgs = _conversation(10)
    out = compact_messages(msgs, keep_recent_tool_results=3)

    tool_msgs = [m for m in out if m.role == "tool"]
    assert len(tool_msgs) == 10  # no messages dropped
    # the last 3 keep their real content; earlier ones are elided
    assert [m.content == ELIDED for m in tool_msgs] == [True] * 7 + [False] * 3
    assert "big file contents 9" in (tool_msgs[-1].content or "")


def test_structure_and_pairing_preserved() -> None:
    msgs = _conversation(5)
    out = compact_messages(msgs, keep_recent_tool_results=1)

    # Same length, same roles in the same order (API pairing intact).
    assert [m.role for m in out] == [m.role for m in msgs]
    # Every elided tool message still carries its tool_call_id.
    for m in out:
        if m.role == "tool":
            assert m.tool_call_id is not None


def test_does_not_mutate_originals() -> None:
    msgs = _conversation(5)
    original_first_tool_content = msgs[3].content  # first tool result
    compact_messages(msgs, keep_recent_tool_results=1)
    assert msgs[3].content == original_first_tool_content  # untouched


def test_zero_keeps_nothing_in_full() -> None:
    msgs = _conversation(4)
    out = compact_messages(msgs, keep_recent_tool_results=0)
    assert all(m.content == ELIDED for m in out if m.role == "tool")
