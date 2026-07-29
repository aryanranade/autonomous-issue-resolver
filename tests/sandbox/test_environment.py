"""Tests for SWEBenchEnvironment (Phase 3c-i).

The image-name test is pure (needs only the swebench package). The provisioning
test needs a docker daemon *and* the instance image already pulled locally — it
skips otherwise, since auto-pulling a multi-GB image inside the suite would be
hostile. Pull it once (live validation) and this test exercises the real flow.
"""

from __future__ import annotations

import subprocess

import pytest

from swe_agent.dataset import SWEBenchInstance
from swe_agent.sandbox.docker import docker_available
from swe_agent.sandbox.environment import SWEBenchEnvironment, instance_image_ref

# A small, pure-Python validation instance (pytest, fast env).
FLASK_COMMIT = "d8c37f43724cd9fb0870f77877b7c4c7e38a19e0"


def _flask_instance() -> SWEBenchInstance:
    return SWEBenchInstance(
        instance_id="pallets__flask-4045",
        repo="pallets/flask",
        base_commit=FLASK_COMMIT,
        problem_statement="Blueprints with dotted names should be rejected.",
        patch="",
        test_patch="",
        fail_to_pass=["tests/test_blueprints.py::test_dotted_name_not_allowed"],
        pass_to_pass=["tests/test_blueprints.py::test_dotted_names"],
        environment_setup_commit=FLASK_COMMIT,
        version="2.0",
    )


def _image_present(ref: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", ref], capture_output=True
        ).returncode
        == 0
    )


def test_instance_image_ref_uses_swebench_namespace_and_mangling() -> None:
    # __ becomes _1776_ and the swebench/ namespace makes it pullable from Hub.
    ref = instance_image_ref(_flask_instance())
    assert ref == "swebench/sweb.eval.x86_64.pallets_1776_flask-4045:latest"


@pytest.mark.skipif(not docker_available(), reason="docker daemon not available")
def test_provision_flask_environment_when_image_present() -> None:
    instance = _flask_instance()
    ref = instance_image_ref(instance)
    if not _image_present(ref):
        pytest.skip(f"instance image not pulled locally: {ref}")

    with SWEBenchEnvironment(instance) as env:
        # The repo was copied onto the host and is mounted into the container.
        assert (env.root / "setup.py").exists() or (env.root / "setup.cfg").exists()

        # HEAD is the base commit (the buggy state the agent must fix).
        head = env.sandbox.exec("git rev-parse HEAD")
        assert head.stdout.strip() == instance.base_commit

        # The conda env is active through the executor: `python` is the testbed
        # interpreter and the package under test imports.
        res = env.executor.run(
            "python -c 'import flask; print(flask.__version__)'",
            cwd=env.root,
            timeout=120,
        )
        assert res.exit_code == 0, res.stderr


def test_repo_test_command_matches_the_official_runner_per_repo() -> None:
    """Only 93/300 SWE-bench Lite instances use pytest; the rest must not.

    django -> ./tests/runtests.py, sympy -> bin/test, sphinx -> tox. Reading the
    command from swebench's own table keeps the agent's verify step aligned with
    how the instance is actually graded.
    """
    from swe_agent.dataset import SWEBenchInstance
    from swe_agent.sandbox.environment import repo_test_command

    def inst(repo: str, version: str) -> SWEBenchInstance:
        return SWEBenchInstance(
            instance_id="x", repo=repo, base_commit="c", problem_statement="p",
            patch="", test_patch="", fail_to_pass=[], pass_to_pass=[],
            environment_setup_commit="c", version=version,
        )

    assert "runtests.py" in (repo_test_command(inst("django/django", "3.0")) or "")
    assert "bin/test" in (repo_test_command(inst("sympy/sympy", "1.0")) or "")
    assert "tox" in (repo_test_command(inst("sphinx-doc/sphinx", "4.1")) or "")
    assert "pytest" in (repo_test_command(inst("pallets/flask", "2.0")) or "")


def test_repo_test_command_returns_none_for_unknown_repo() -> None:
    """Unknown repo/version must fall back to the pytest default, not guess."""
    from swe_agent.dataset import SWEBenchInstance
    from swe_agent.sandbox.environment import repo_test_command

    unknown = SWEBenchInstance(
        instance_id="x", repo="not/a-real-repo", base_commit="c",
        problem_statement="p", patch="", test_patch="", fail_to_pass=[],
        pass_to_pass=[], environment_setup_commit="c", version="9.9",
    )
    assert repo_test_command(unknown) is None
