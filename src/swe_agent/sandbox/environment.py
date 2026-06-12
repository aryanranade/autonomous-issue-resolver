"""Provision a ready-to-solve sandbox for one SWE-bench instance (Phase 3c-i).

SWE-bench ships a prebuilt Docker image per instance: the repository checked out
at ``/testbed`` with its full dependency stack in a conda env named ``testbed``.
The agent needs to edit that repo and run its tests. Two facts from earlier
phases shape the design:

* Our file tools (read_file / edit_file / list_dir / search_code) are
  **host-side** — they touch ``ToolContext.root`` directly. Only shell and test
  commands go through the executor. So the host and the container must see the
  *same* repo files.
* Bind-mounting a host directory at ``/testbed`` would *shadow* the image's
  prebuilt repo (and its in-place editable install).

So we copy ``/testbed`` out of the image into a host scratch directory, then
bind-mount that directory back at ``/testbed``. The mount path is unchanged, so
the image's ``pip install -e .`` (pinned to ``/testbed``) keeps working, while
the agent edits the host copy and the container's pytest sees those edits.

This host is arm64 and the instance images are x86_64-only, so we run them under
emulation (``--platform linux/amd64``). Image names come from the ``swebench``
package (the authoritative source), never hand-rolled strings.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from swe_agent.dataset import SWEBenchInstance
from swe_agent.sandbox.docker import DockerExecutor, DockerSandbox
from swe_agent.tools.base import ToolContext

logger = logging.getLogger(__name__)

WORKDIR = "/testbed"
# Run x86_64 instance images emulated on this arm64 host.
PLATFORM = "linux/amd64"
# Activate the image's conda env so `python` / `pytest` are the testbed ones.
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"
# Host scratch lives under $HOME so Docker Desktop / colima reliably share it
# into the VM (the macOS default temp dir under /var/folders often isn't shared).
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "swe-agent"


def _instance_to_row(instance: SWEBenchInstance) -> dict[str, Any]:
    """Re-shape a SWEBenchInstance into the dict ``make_test_spec`` consumes."""
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "version": instance.version,
        "base_commit": instance.base_commit,
        "environment_setup_commit": instance.environment_setup_commit,
        "problem_statement": instance.problem_statement,
        "patch": instance.patch,
        "test_patch": instance.test_patch,
        "FAIL_TO_PASS": instance.fail_to_pass,
        "PASS_TO_PASS": instance.pass_to_pass,
    }


def make_spec(instance: SWEBenchInstance) -> Any:
    """Return swebench's ``TestSpec`` for an instance.

    The TestSpec carries the image keys, the eval script, and the gold test
    lists. Both provisioning (image name) and grading (eval script + parser)
    rely on it, so it's the single authoritative bridge to swebench. Returns
    ``Any`` because swebench ships no type stubs.
    """
    from swebench.harness.test_spec.test_spec import (  # type: ignore[import-untyped]
        make_test_spec,
    )

    return make_test_spec(_instance_to_row(instance), namespace="swebench")


def instance_image_ref(instance: SWEBenchInstance) -> str:
    """Return the pullable Docker image for an instance (x86_64, swebench namespace).

    e.g. ``pallets__flask-4045`` ->
    ``swebench/sweb.eval.x86_64.pallets_1776_flask-4045:latest``.

    swebench owns this naming (including the ``__`` -> ``_1776_`` tag mangling),
    so we ask it rather than format the string ourselves — robust to scheme
    changes and guaranteed to match the images actually published on Docker Hub.
    """
    return str(make_spec(instance).instance_image_key)


class SWEBenchEnvironment:
    """A running container for one instance, ready for the agent to work in.

    Use as a context manager::

        with SWEBenchEnvironment(instance) as env:
            agent = Agent(llm, registry, env.tool_context(), config)
            result = agent.run(instance.to_task())
            patch = result.patch  # unified diff against base_commit

    On enter: copy the image's ``/testbed`` to a host scratch dir, bind-mount it
    back at ``/testbed`` under emulation, and reset the repo to ``base_commit`` so
    the agent starts from the buggy state and the captured diff is base-relative.
    On exit: remove the container and the scratch dir.
    """

    def __init__(
        self,
        instance: SWEBenchInstance,
        *,
        cache_root: Path | None = None,
        platform: str = PLATFORM,
    ) -> None:
        self.instance = instance
        self.image = instance_image_ref(instance)
        self.platform = platform
        self._run_dir = (cache_root or DEFAULT_CACHE_ROOT) / (
            f"{instance.instance_id}-{uuid.uuid4().hex[:8]}"
        )
        self._host_testbed = self._run_dir / "testbed"
        self._sandbox: DockerSandbox | None = None
        self._executor: DockerExecutor | None = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> "SWEBenchEnvironment":
        """Pull/extract the image, mount the repo, and reset it to base_commit."""
        self._host_testbed.mkdir(parents=True, exist_ok=True)
        self._populate_host_testbed()

        sandbox = DockerSandbox(
            self.image,
            workdir=WORKDIR,
            platform=self.platform,
            mounts=[(str(self._host_testbed), WORKDIR)],
        )
        sandbox.start()
        self._sandbox = sandbox
        self._executor = DockerExecutor(sandbox, command_prefix=CONDA_ACTIVATE)
        self._reset_to_base_commit()
        logger.info(
            "environment ready for %s (%s) at %s",
            self.instance.instance_id,
            self.image,
            self._host_testbed,
        )
        return self

    def _populate_host_testbed(self) -> None:
        """Copy the image's ``/testbed`` onto the host so we can mount it back.

        A bind mount shadows image content, so we extract first via a throwaway
        container, then mount the now-populated host directory.
        """
        with DockerSandbox(self.image, platform=self.platform) as tmp:
            tmp.copy_out(f"{WORKDIR}/.", self._host_testbed)

    def _reset_to_base_commit(self) -> None:
        """Force the repo to ``base_commit`` (detached) so the agent sees the bug.

        ``safe.directory`` is required because the bind-mounted tree is owned by
        the host user, not the container's root — git would otherwise refuse to
        operate on it ("dubious ownership"). A forced checkout discards any drift
        from the image's setup commit and leaves HEAD at base_commit, so the
        diff captured after the agent runs is exactly its own changes.
        """
        result = self.sandbox.exec(
            "git config --global --add safe.directory /testbed && "
            f"git checkout -f {self.instance.base_commit}",
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to reset {self.instance.instance_id} to "
                f"{self.instance.base_commit}: {result.stderr.strip()}"
            )

    def stop(self) -> None:
        """Remove the container and delete the host scratch dir. Idempotent."""
        if self._sandbox is not None:
            self._sandbox.stop()
            self._sandbox = None
        self._executor = None
        shutil.rmtree(self._run_dir, ignore_errors=True)

    # ---- accessors ---------------------------------------------------------

    @property
    def sandbox(self) -> DockerSandbox:
        if self._sandbox is None:
            raise RuntimeError("environment not started; call start() first")
        return self._sandbox

    @property
    def executor(self) -> DockerExecutor:
        if self._executor is None:
            raise RuntimeError("environment not started; call start() first")
        return self._executor

    @property
    def root(self) -> Path:
        """Host path the agent's file tools operate on (mounted at /testbed)."""
        return self._host_testbed

    def tool_context(self, **kwargs: Any) -> ToolContext:
        """A ToolContext wired to this environment: host root + docker executor.

        ``python_executable="python"`` so run_tests invokes the activated conda
        env inside the container rather than this host's interpreter.
        """
        return ToolContext(
            root=self.root,
            executor=self.executor,
            python_executable="python",
            **kwargs,
        )

    def __enter__(self) -> "SWEBenchEnvironment":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
