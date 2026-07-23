"""Load and score the vendored MBPP+ subset (the v2 code study's benchmark).

This is the code analogue of ``multi_constraint``: it turns a committed dataset into
``BenchmarkTask``s and supplies the ``score``/``feedback`` the study harness injects.
The mechanism thesis being tested is that loop-level GEPA (arm D) should beat prompt-only
(arm B) where verification is *reliable and executable* — which code satisfies via unit
tests, unlike v1's LLM self-critique. See the 2026-07-23 code pre-registration.

The honesty crux is an anti-oracle split (from EvalPlus, plus ⊇ base):
- **base_tests** (visible) — shown in the worker prompt and executed by the in-loop
  critique (``code_feedback`` / the loop's ``exec_feedback``). This is the realistic,
  executable signal the loop may optimize against.
- **plus_test** (hidden) — the held-out scorer (``code_score``). Its plus-only edge cases
  are never shown to the worker, so passing base does NOT guarantee passing plus (that gap
  is the headroom EvalPlus exposes). The programmatic scorer is never consulted in-loop.

Data is loaded ONLY from the committed JSON (see ``scripts/vendor_mbpp_plus.py``): no
network, no ``datasets``/``evalplus`` dependency, and every reference solution was verified
to pass both suites at vendoring time. Scoring executes untrusted model code through the
sandbox (``code_exec.run_tests``) and is deterministic (pure functions, pinned env), so
held-out scores replay bit-for-bit.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.code_exec import run_tests

_DATA = Path(__file__).parent / "data" / "mbpp_plus.json"

# A fenced code block: ```optional-lang \n ... \n ```. DOTALL so the body spans lines.
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the Python out of a worker's answer.

    Models wrap code in Markdown fences (often with a language tag, sometimes several
    blocks — imports separate from the function). We concatenate every fenced block so a
    split answer stays runnable; with no fence we treat the whole text as raw code. A best
    effort by design: a solution that never yields runnable code simply fails its tests
    (a NameError on the entry point), which is the correct 0.0 — not a crash."""
    blocks = _FENCE.findall(text)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks)
    return text


def _payload(task: BenchmarkTask) -> dict[str, Any]:
    """The task's code payload, or a loud error if this isn't a code task.

    A missing field means the study was wired with the wrong loader/scorer pairing; a
    silent 0.0 would bury that misconfiguration in a paid report as an all-fail genome."""
    payload = task.payload or {}
    for key in ("plus_test", "base_tests", "entry_point"):
        if key not in payload:
            raise KeyError(
                f"task {task.id!r} has no {key!r}; code_score/code_feedback need a code "
                "task from load_code_tasks (wrong task-set / scorer wiring?)"
            )
    return payload


def _base_block(payload: dict[str, Any]) -> str:
    return "\n".join(payload["base_tests"])


def code_score(output: str, task: BenchmarkTask) -> float:
    """Held-out score: 1.0 iff the extracted solution passes the HIDDEN plus suite.

    Binary because the plus harness is an all-or-nothing assertion loop. The worker never
    sees ``plus_test``, so this cannot be gamed from inside the loop."""
    payload = _payload(task)
    return 1.0 if run_tests(extract_code(output), payload["plus_test"]).passed else 0.0


def code_feedback(output: str, task: BenchmarkTask) -> str:
    """In-loop / reflection signal from the VISIBLE base tests only (never the plus suite).

    Used both as the loop's execution feedback (``exec_feedback``) and as GEPA's reflection
    prose (``feedback_fn``). Reporting base results only keeps the hidden scorer out of both
    the loop and the optimizer. Honest wording: passing base is necessary, not sufficient —
    plus-only edge cases can still fail, so we never claim the answer is correct."""
    payload = _payload(task)
    result = run_tests(extract_code(output), _base_block(payload))
    n = len(payload["base_tests"])
    if result.passed:
        return f"All {n} visible base test(s) pass (plus-only edge cases may still fail)."
    return f"A visible base test failed:\n{result.detail}"


def check_satisfiable(task: BenchmarkTask) -> bool:
    """True iff the stored reference solution passes BOTH the base and plus suites.

    A guard against a corrupted/edited data file — the vendoring script enforces this, and
    a study run can assert it before spending."""
    payload = _payload(task)
    ref = payload.get("reference", "")
    return (
        bool(ref)
        and run_tests(ref, _base_block(payload)).passed
        and run_tests(ref, payload["plus_test"]).passed
    )


def _render_goal(prompt: str, entry_point: str, base_tests: list[str]) -> str:
    """The instruction shown to the worker: the task + the entry-point name + the VISIBLE
    base tests. The plus suite and the reference are deliberately omitted."""
    tests = "\n".join(base_tests)
    return (
        f"Write a Python function named `{entry_point}` that solves the following task.\n\n"
        f"{prompt}\n\n"
        f"Your function must pass these tests:\n{tests}\n\n"
        "Respond with the function definition only (a ```python code block is fine)."
    )


@lru_cache(maxsize=1)
def vendored_tasks() -> tuple[dict[str, Any], ...]:
    """The committed MBPP+ subset as raw dicts (cached; read once from the JSON)."""
    doc = json.loads(_DATA.read_text(encoding="utf-8"))
    return tuple(doc["tasks"])


def _to_benchmark_task(raw: dict[str, Any]) -> BenchmarkTask:
    return BenchmarkTask(
        id=f"mbpp-{raw['task_id']}",
        payload={
            "goal": _render_goal(raw["prompt"], raw["entry_point"], raw["base_tests"]),
            "entry_point": raw["entry_point"],
            "base_tests": raw["base_tests"],
            "plus_test": raw["plus_test"],
            "reference": raw["reference"],
        },
    )


Split = Literal["all", "train", "heldout"]


def load_code_tasks(
    split: Split = "all", *, n: int | None = None, seed: int = 0
) -> list[BenchmarkTask]:
    """Deterministically shuffle the vendored pool and split it into disjoint halves.

    Mirrors ``load_multi_constraint``: same seed → same tasks; ``train``/``heldout`` are
    the first/second half of the shuffled pool (disjoint); ``n`` caps the result and raises
    rather than silently truncating below the requested (and cost-priced) sample size."""
    tasks = [_to_benchmark_task(r) for r in vendored_tasks()]
    random.Random(seed).shuffle(tasks)
    if split != "all":
        mid = len(tasks) // 2
        tasks = tasks[:mid] if split == "train" else tasks[mid:]
    if n is not None:
        if n > len(tasks):
            raise ValueError(
                f"requested n={n} {split!r} code tasks but only {len(tasks)} available "
                f"(pool={len(vendored_tasks())}); lower n or vendor a larger pool — silent "
                "truncation would make the reported sample size disagree with what ran"
            )
        tasks = tasks[:n]
    return tasks
