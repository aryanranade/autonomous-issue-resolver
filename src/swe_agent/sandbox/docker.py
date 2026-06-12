"""Docker-backed sandbox: run commands inside an isolated container.

This is the Phase 3 payoff of the Phase 1 design: ``DockerExecutor`` implements
the same ``CommandExecutor`` interface as ``LocalExecutor``, so the agent and
tools run unchanged — only the executor swaps. We shell out to the ``docker``
CLI (no Python docker SDK dependency), consistent with how ``LocalExecutor``
uses ``subprocess``.

A ``DockerSandbox`` owns one container's lifecycle; a ``DockerExecutor`` adapts
it to the executor interface. Commands always run in the container's workdir
(where the repo lives), so the host ``cwd`` the tools pass is ignored.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path

from swe_agent.tools.base import CommandExecutor, ExecResult

logger = logging.getLogger(__name__)

_DOCKER = "docker"


class DockerError(RuntimeError):
    """Raised when a docker CLI invocation fails unexpectedly."""


def docker_available() -> bool:
    """True if a reachable docker daemon is present (used to skip tests/guard CLI)."""
    try:
        proc = subprocess.run(
            [_DOCKER, "info"], capture_output=True, timeout=20
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode(errors="replace")


class DockerSandbox:
    """Manages a single container: start, exec, copy files out, tear down."""

    def __init__(
        self,
        image: str,
        *,
        workdir: str = "/testbed",
        platform: str | None = None,
        mounts: Sequence[tuple[str, str]] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            image: image reference to run.
            workdir: default working directory inside the container.
            platform: e.g. "linux/amd64" to force emulation for x86-only images
                on this arm64 host. None means use the image's native platform.
            mounts: (host_path, container_path) bind mounts (used in Phase 3c to
                expose the repo so host-side file tools and in-container tests
                share the same files).
            name: container name; a unique one is generated if omitted.
        """
        self.image = image
        self.workdir = workdir
        self.platform = platform
        self.mounts = list(mounts or [])
        self.name = name or f"swe-agent-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None

    def start(self, *, timeout: int = 900) -> str:
        """Start the container (pulling the image if needed) and keep it alive.

        Uses ``sleep infinity`` as the command so the container stays up for
        repeated ``exec`` calls. ``timeout`` is generous to allow image pulls.
        """
        if self.container_id is not None:
            return self.container_id
        args = [_DOCKER, "run", "-d", "--name", self.name]
        if self.platform:
            args += ["--platform", self.platform]
        for host_path, container_path in self.mounts:
            args += ["-v", f"{Path(host_path).resolve()}:{container_path}"]
        args += [self.image, "sleep", "infinity"]

        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise DockerError(
                f"failed to start container from {self.image!r}: {proc.stderr.strip()}"
            )
        self.container_id = proc.stdout.strip()
        logger.info("started container %s from %s", self.container_id[:12], self.image)
        return self.container_id

    def exec(
        self,
        command: str,
        *,
        workdir: str | None = None,
        timeout: int = 300,
        shell: str = "sh",
    ) -> ExecResult:
        """Run ``command`` (via ``<shell> -c``) inside the container.

        ``shell`` defaults to ``sh``; pass ``bash`` for commands that need bash
        features such as ``source`` (e.g. activating a conda env).
        """
        if self.container_id is None:
            raise DockerError("sandbox not started; call start() first")
        args = [
            _DOCKER, "exec", "-w", workdir or self.workdir,
            self.container_id, shell, "-c", command,
        ]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr) + f"\n[timed out after {timeout}s]",
                timed_out=True,
            )
        return ExecResult(
            exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )

    def copy_out(self, container_src: str, host_dest: Path | str) -> None:
        """Copy a path from the container to the host (``docker cp``)."""
        if self.container_id is None:
            raise DockerError("sandbox not started; call start() first")
        proc = subprocess.run(
            [_DOCKER, "cp", f"{self.container_id}:{container_src}", str(host_dest)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise DockerError(f"docker cp failed: {proc.stderr.strip()}")

    def stop(self) -> None:
        """Force-remove the container. Safe to call multiple times."""
        if self.container_id is None:
            return
        subprocess.run(
            [_DOCKER, "rm", "-f", self.container_id],
            capture_output=True, text=True,
        )
        logger.info("removed container %s", self.container_id[:12])
        self.container_id = None

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class DockerExecutor(CommandExecutor):
    """Runs commands in a DockerSandbox, satisfying the CommandExecutor seam.

    The host ``cwd`` argument is intentionally ignored: the repository lives at
    a fixed path inside the container, so all commands run in the sandbox's
    workdir.

    ``command_prefix`` is prepended (``<prefix> && <command>``) to every command.
    Phase 3c uses it to activate the SWE-bench image's conda env so ``python`` /
    ``pytest`` resolve to the task's environment rather than the bare container's.
    """

    def __init__(
        self,
        sandbox: DockerSandbox,
        *,
        workdir: str | None = None,
        command_prefix: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._workdir = workdir or sandbox.workdir
        self._command_prefix = command_prefix

    def run(self, command: str, *, cwd: Path, timeout: int) -> ExecResult:
        if self._command_prefix:
            # The prefix (conda activation) uses bash-only `source`, so the whole
            # command must run under bash rather than the default sh.
            command = f"{self._command_prefix} && {command}"
            return self._sandbox.exec(
                command, workdir=self._workdir, timeout=timeout, shell="bash"
            )
        return self._sandbox.exec(command, workdir=self._workdir, timeout=timeout)
