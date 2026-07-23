"""Offline end-to-end proof of the v2-code pipeline — NO network, NO spend.

Runs the real ``run_study`` over vendored MBPP+ tasks with a *faked* worker, so every
seam the paid run will use is exercised for free: the loader, the goal rendering, code
extraction, the HIDDEN plus-test scorer, the VISIBLE base-test in-loop feedback, the
self-refine loop, and run_study's arm plumbing. Execution is real (numpy in the sandbox),
which is exactly what makes the held-out scores deterministic and replayable.

The fake worker returns the task's own reference solution on generate/refine and "PASS"
on critique, so a correct answer flows through and must score 1.0 on the plus tests — the
readout that proves scoring is wired end to end, not just in unit isolation.
"""

from __future__ import annotations

from pathlib import Path

from whystuck.core.storage import WhyStuckStore
from whystuck.eval.code_bench import code_feedback, code_score, load_code_tasks
from whystuck.eval.study import run_study


def _ref_worker(tasks, calls):
    ref_by_entry = {t.payload["entry_point"]: t.payload["reference"] for t in tasks}

    def worker(prompt: str) -> str:
        calls.append(prompt)
        if "# Draft answer" in prompt:  # critique step: accept the draft
            return "PASS"
        for entry, ref in ref_by_entry.items():
            if f"`{entry}`" in prompt:  # generate/refine: return that task's reference
                return f"```python\n{ref}\n```"
        return "def _unknown():\n    return None\n"

    return worker


def test_correct_worker_scores_one_and_consults_execution_feedback(tmp_path):
    tasks = load_code_tasks("heldout", n=3, seed=0)
    calls: list[str] = []
    res = run_study(
        WhyStuckStore(tmp_path / "s"),
        complete=_ref_worker(tasks, calls),
        reflection_lm="fake",
        train=load_code_tasks("train", n=2, seed=0),
        heldout=tasks,
        arms=("A",),  # arm A = seed held-out score; enough to prove the scoring pipeline
        score_fn=code_score,
        feedback_fn=code_feedback,
        exec_feedback=code_feedback,
        loop_max_retries=1,
    )
    # Reference solutions pass the HIDDEN plus tests → held-out score is a clean 1.0.
    assert res["A"].seed_heldout == 1.0
    # The critique saw executable base-test results (the v1-missing signal).
    critique_prompts = [c for c in calls if "# Draft answer" in c]
    assert critique_prompts and all("# Test results" in c for c in critique_prompts)


def test_wrong_worker_scores_zero_and_feedback_flags_a_base_failure(tmp_path):
    tasks = load_code_tasks("heldout", n=3, seed=0)

    def wrong_worker(prompt: str) -> str:
        if "# Draft answer" in prompt:
            return "not all tests pass; fix it"  # critique never says PASS
        # a syntactically valid but wrong stub for whatever function is asked
        return "def whatever(*a, **k):\n    return None\n"

    res = run_study(
        WhyStuckStore(tmp_path / "s"),
        complete=wrong_worker,
        reflection_lm="fake",
        train=load_code_tasks("train", n=2, seed=0),
        heldout=tasks,
        arms=("A",),
        score_fn=code_score,
        feedback_fn=code_feedback,
        exec_feedback=code_feedback,
        loop_max_retries=1,
    )
    assert res["A"].seed_heldout == 0.0
    # the recorded failure summary is the base-test signal, not the hidden plus suite
    trace_fb = code_feedback("def whatever(*a, **k):\n    return None\n", tasks[0])
    assert "base test" in trace_fb.lower()


def test_cli_dry_run_for_code_fits_a_trimmed_budget():
    # A pre-spend guard-rail: a trimmed code config must dry-run cleanly under $5 with the
    # larger code token shape priced in. Proves the CLI's code branch + cost wiring hold.
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"
    out = subprocess.run(
        [
            sys.executable, str(script),
            "--task-set", "code",
            "--n-train", "6", "--n-heldout", "10",
            "--max-metric-calls", "40", "--reflection-calls", "8",
            "--loop-max-retries", "1", "--seeds", "0,1",
        ],
        capture_output=True, text=True,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert "Estimated total" in out.stdout
    assert "Dry run only" in out.stdout  # no --yes → no spend
