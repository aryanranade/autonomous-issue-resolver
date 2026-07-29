"""Tests for the Docker sandbox.

These require a working docker daemon and are skipped automatically otherwise,
so the rest of the suite still runs on machines without Docker. They use the
small ``python:3.11-slim`` image.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from swe_agent.sandbox.docker import (
    DockerExecutor,
    DockerSandbox,
    docker_available,
)
from swe_agent.tools.base import ToolContext
from swe_agent.tools.registry import default_registry

pytestmark = pytest.mark.skipif(
    not docker_available(), reason="docker daemon not available"
)

IMAGE = "python:3.11-slim"


@pytest.fixture(scope="module")
def sandbox() -> Iterator[DockerSandbox]:
    """One shared, started container for the read-only exec tests."""
    sb = DockerSandbox(IMAGE, workdir="/")
    sb.start()
    try:
        yield sb
    finally:
        sb.stop()


def test_runs_command_and_captures_stdout(sandbox: DockerSandbox) -> None:
    result = sandbox.exec("python --version")
    assert result.exit_code == 0
    assert "Python 3.11" in (result.stdout + result.stderr)


def test_nonzero_exit_is_captured(sandbox: DockerSandbox) -> None:
    result = sandbox.exec("exit 7")
    assert result.exit_code == 7
    assert result.timed_out is False


def test_command_runs_in_workdir(sandbox: DockerSandbox) -> None:
    result = sandbox.exec("pwd", workdir="/tmp")
    assert result.stdout.strip() == "/tmp"


def test_timeout_is_reported(sandbox: DockerSandbox) -> None:
    result = sandbox.exec("sleep 5", timeout=1)
    assert result.timed_out is True
    assert result.exit_code == 124


def test_isolation_writes_dont_touch_host(sandbox: DockerSandbox) -> None:
    """A write inside the container must not appear on the host at the same path.

    Uses a unique name under /tmp rather than /root: /tmp is readable on every
    host (stat'ing /root raises PermissionError on Linux CI runners), and the
    random name means a pre-existing file can't make this pass by accident.
    """
    sentinel = f"/tmp/only_in_container_{uuid.uuid4().hex}.txt"

    assert not Path(sentinel).exists(), "host path must be clean before the write"

    sandbox.exec(f"echo hi > {sentinel}")

    # Present inside the container...
    assert sandbox.exec(f"cat {sentinel}").stdout.strip() == "hi"
    # ...and still absent on the host: that's the isolation guarantee.
    assert not Path(sentinel).exists()


def test_context_manager_removes_container() -> None:
    import subprocess

    name = None
    # workdir="/" because the default /testbed only exists in SWE-bench images.
    with DockerSandbox(IMAGE, workdir="/", name="swe-agent-cm-test") as sb:
        name = sb.name
        assert sb.exec("true").exit_code == 0
    # After exit the container should be gone.
    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    assert name not in listed.stdout


def test_docker_executor_satisfies_command_executor_seam(sandbox: DockerSandbox) -> None:
    # The whole point: drive a Phase 1 tool through the registry, but executing
    # inside the container via DockerExecutor instead of LocalExecutor.
    executor = DockerExecutor(sandbox, workdir="/tmp")
    ctx = ToolContext(root=Path("/tmp"), executor=executor)
    result = default_registry().dispatch("run_shell", {"command": "uname -s"}, ctx)
    assert result.ok
    assert "Linux" in result.output  # we're in the container, not on macOS


def test_copy_out_extracts_file(sandbox: DockerSandbox, tmp_path: Path) -> None:
    sandbox.exec("echo payload > /tmp/extract_me.txt")
    dest = tmp_path / "extracted.txt"
    sandbox.copy_out("/tmp/extract_me.txt", dest)
    assert dest.read_text().strip() == "payload"
