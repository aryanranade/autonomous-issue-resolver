"""The unit of work the agent operates on.

Minimal on purpose. Phase 3 (SWE-bench loading) will carry more per-task
metadata (repo, base commit, test command, gold patch); this is the slice the
agent loop itself needs: an id and the problem statement to fix.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    problem_statement: str
