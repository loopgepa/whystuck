"""Offline end-to-end proof of the BigCodeBench (Docker) pipeline — NO network, NO spend.

Runs the real ``run_study`` over vendored BigCodeBench-Hard tasks with a faked worker, so
every seam the paid run uses is exercised for free: the loader, goal rendering, code
extraction, the Docker-sandboxed HIDDEN-method scorer, the VISIBLE-method in-loop feedback,
the self-refine loop, and run_study's plumbing. Execution is real (inside the whystuck-bcb
container), which is what makes the held-out scores deterministic and replayable.

Skips without Docker + the image + the vendored data. Slow (real containers) but small (n=2).
"""

from __future__ import annotations

import os
import subprocess

import pytest

from whystuck.core.storage import WhyStuckStore
from whystuck.eval.bcb_bench import _DATA, bcb_feedback, bcb_score, load_bcb_tasks
from whystuck.eval.study import run_study


def _docker_ok() -> bool:
    env = {
        **os.environ,
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
    }
    try:
        for args in (["docker", "info"], ["docker", "image", "inspect", "whystuck-bcb:latest"]):
            if subprocess.run(
                args, capture_output=True, timeout=15, env=env, check=False
            ).returncode:
                return False
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not (_docker_ok() and _DATA.exists()),
    reason="needs Docker + whystuck-bcb image + vendored bigcodebench_hard.json",
)


def test_reference_worker_scores_one_and_consults_visible_execution(tmp_path):
    tasks = load_bcb_tasks("heldout", n=2, seed=0)
    ref_by_prompt = [(t.payload["reference"], t.payload["goal"]) for t in tasks]
    calls: list[str] = []

    def worker(prompt: str) -> str:
        calls.append(prompt)
        if "# Draft answer" in prompt:  # critique step
            return "PASS"
        for ref, goal in ref_by_prompt:  # generate/refine: return the matching reference
            spec = goal.split("```python code block.\n\n", 1)[-1]
            if spec[:200] in prompt:
                return ref
        return "def task_func(*a, **k):\n    return None\n"

    res = run_study(
        WhyStuckStore(tmp_path / "s"),
        complete=worker,
        reflection_lm="fake",
        train=load_bcb_tasks("train", n=2, seed=0),
        heldout=tasks,
        arms=("A",),
        score_fn=bcb_score,
        feedback_fn=bcb_feedback,
        exec_feedback=bcb_feedback,
        loop_max_retries=1,
    )
    assert res["A"].seed_heldout == 1.0  # references pass the hidden methods in the sandbox
    critique_prompts = [c for c in calls if "# Draft answer" in c]
    assert critique_prompts and all("# Test results" in c for c in critique_prompts)


def test_wrong_worker_scores_zero(tmp_path):
    tasks = load_bcb_tasks("heldout", n=2, seed=0)

    def wrong(prompt: str) -> str:
        if "# Draft answer" in prompt:
            return "still failing"
        return "def task_func(*a, **k):\n    return None\n"

    res = run_study(
        WhyStuckStore(tmp_path / "s"),
        complete=wrong,
        reflection_lm="fake",
        train=load_bcb_tasks("train", n=2, seed=0),
        heldout=tasks,
        arms=("A",),
        score_fn=bcb_score,
        feedback_fn=bcb_feedback,
        exec_feedback=bcb_feedback,
        loop_max_retries=1,
    )
    assert res["A"].seed_heldout == 0.0
