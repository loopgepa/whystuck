"""Shared test constants and factories.

Kills the timestamp literal that was hardcoded across the suite and the
near-duplicate trace/genome/proposal builders that had drifted between modules.
Import from here: ``from conftest import FIXED_TS, trace_dict, rubric_genome``.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any

import pytest

from whystuck.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    MutationProposal,
    MutationType,
)

# One fixed instant so tests never depend on the wall clock.
FIXED_TS = "2026-07-20T00:00:00Z"


@pytest.fixture
def hide_import(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Make `import NAME` raise, so a test of the missing-dependency path really tests it.

    The eval extra (gepa, litellm) is OPTIONAL by ADR-011, and two tests asserting that its
    absence produces an *actionable* error asserted nothing for as long as the extra happened to
    be uninstalled in the dev venv. When it was reinstalled they failed --- not because the error
    message regressed, but because the import they expected to fail succeeded and the call ran on
    into the real library. A test of an error path has to create the error; inferring it from the
    environment means the assertion is only as good as today's `uv sync`.

    Patching `builtins.__import__` rather than poking `sys.modules` is deliberate: it blocks the
    submodule forms too (`from gepa.core.adapter import ...`) without depending on which parent
    packages a previous test already imported.
    """
    real = builtins.__import__
    hidden: set[str] = set()

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if any(name == h or name.startswith(f"{h}.") for h in hidden):
            raise ImportError(f"hidden by the hide_import fixture: {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    def hide(*names: str) -> None:
        hidden.update(names)

    return hide


def trace_dict(tid: str, outcome: str = "failure", summary: str | None = "boom") -> dict[str, Any]:
    """A LoopTrace as a plain dict, for handlers that validate raw input."""
    return {
        "id": tid,
        "loop_id": "bugfix",
        "created_at": FIXED_TS,
        "final_outcome": outcome,
        "failure_summary": summary,
    }


def rubric_genome() -> LoopGenome:
    """A minimal loop: one verifier_rubric gene."""
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
    )


def edit_proposal(target: str, after: str | dict[str, Any], mid: str = "m1") -> MutationProposal:
    """An edit_gene MutationProposal citing one trace."""
    return MutationProposal(
        id=mid,
        loop_id="bugfix",
        mutation_type=MutationType.EDIT_GENE,
        target_gene_id=target,
        title="edit",
        rationale="because",
        evidence_trace_ids=["t1"],
        after=after,
    )


# --------------------------------------------------------------- the sandbox image guard
#
# The BigCodeBench tests execute real solutions inside a PINNED Docker image built from
# `docker/bcb/Dockerfile`. It is a LOCAL build, not a pull, so a host that has pruned its images
# no longer has it -- and every one of those tests then fails with `sandbox_error`, eleven
# identical failures that read as "the code is broken" when the truth is "the sandbox is absent".
# That happened on this host and cost a debugging cycle.
#
# Skip in that case, loudly and with the rebuild command, but ONLY for a missing image. Any other
# sandbox fault still fails: a silently skipped security test is worse than a noisy one, and two
# of these tests are the ones asserting a solution cannot fake a pass.

_BCB_IMAGE_STATE: dict[str, str | None] = {}


#: What `docker` prints when it cannot reach a daemon at all, as opposed to reaching one that has
#: never built the image. The two are different findings and used to produce the same skip reason.
_DAEMON_UNREACHABLE = (
    "Is the docker daemon running",
    "failed to connect to the docker API",
    "Cannot connect to the Docker daemon",
    "docker daemon is not running",
)


def _bcb_image_missing() -> str | None:
    """A reason string when the pinned image is unavailable, else None. Probed once per session.

    Probed with the environment the sandbox itself will use, and deliberately without defaulting
    `DOCKER_HOST`: `whystuck.eval.bcb_exec` runs `docker run` with the inherited environment, so
    Docker resolves the daemon through the active *context* there. A probe that overrode the
    variable would be more permissive than the runtime, and the tests would run against a daemon
    the product cannot reach --- which is how this reads on a host whose `desktop-linux` context
    points at a socket that is not there while `unix:///var/run/docker.sock` answers.

    The two failures are told apart rather than merged. On this host the image IS present and the
    context IS dead, and the old message said "the pinned sandbox image is not present", sending a
    reader to rebuild an image that already existed.
    """
    if "reason" in _BCB_IMAGE_STATE:
        return _BCB_IMAGE_STATE["reason"]
    import os
    import shutil
    import subprocess

    image = os.environ.get("WHYSTUCK_BCB_IMAGE", "whystuck-bcb:latest")
    reason: str | None = None
    if shutil.which("docker") is None:
        reason = "docker is not installed on this host"
    else:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 and any(m in proc.stderr for m in _DAEMON_UNREACHABLE):
            reason = (
                "no docker daemon is reachable the way the sandbox reaches one. `docker context "
                "ls` will show which endpoint is active; a context pointing at a stopped Docker "
                "Desktop is the usual cause, and `docker context use default` or "
                "DOCKER_HOST=unix:///var/run/docker.sock is the fix. The image may well be built "
                "already: this is a connection fault, not a missing image"
            )
        elif proc.returncode != 0:
            reason = (
                f"the pinned sandbox image {image!r} is not present on this host. "
                f"Rebuild it (it is fully pinned, so this is reproducible): "
                f"docker build -t {image} docker/bcb/"
            )
        else:
            reason = _probe_mount(image)
    _BCB_IMAGE_STATE["reason"] = reason
    return reason


def _probe_mount(image: str) -> str | None:
    """Confirm the sandbox can bind-mount a workdir, and point it somewhere shareable if not.

    A restricted Docker file-sharing allowlist (Docker Desktop, some rootless setups) refuses to
    mount /tmp and reports a generic `docker run failed`. That is an INFRA fault, but it is
    indistinguishable from a broken image unless the message is read -- and it silently turns
    every sandbox test into a failure that looks like broken code. So: try the default location,
    and on refusal retry under the user's cache directory and set WHYSTUCK_BCB_TMPDIR for the
    session. Only if neither works is the sandbox genuinely unavailable.
    """
    import os
    import subprocess
    import tempfile

    def _can_mount(parent: str | None) -> bool:
        with tempfile.TemporaryDirectory(dir=parent) as d:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "-v",
                    f"{d}:/w",
                    "-w",
                    "/w",
                    image,
                    "python",
                    "-c",
                    "print(1)",
                ],
                capture_output=True,
                text=True,
            )
        return proc.returncode == 0

    if _can_mount(None):
        return None
    fallback = os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "whystuck-bcb-tmp"
    )
    os.makedirs(fallback, exist_ok=True)
    if _can_mount(fallback):
        os.environ["WHYSTUCK_BCB_TMPDIR"] = fallback
        return None
    return (
        "docker cannot bind-mount a sandbox workdir from either the default temp location or "
        f"{fallback!r}. This is a host file-sharing restriction, not a code fault: add one of "
        "those paths to Docker's shared paths, or set WHYSTUCK_BCB_TMPDIR to a path it will "
        "mount."
    )


@pytest.fixture(autouse=True)
def _bcb_sandbox_guard(request):
    """Skip a sandbox-executing test when the image is absent, with the rebuild command."""
    if not any(m in request.node.nodeid for m in ("bcb_exec", "bcb_e2e", "bcb_bench")):
        return
    reason = _bcb_image_missing()
    if reason:
        pytest.skip(reason)
