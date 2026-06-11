"""Execution sandbox (Phase 3): per-task Docker containers for isolation."""

from swe_agent.sandbox.docker import (
    DockerError,
    DockerExecutor,
    DockerSandbox,
    docker_available,
)

__all__ = [
    "DockerError",
    "DockerExecutor",
    "DockerSandbox",
    "docker_available",
]
