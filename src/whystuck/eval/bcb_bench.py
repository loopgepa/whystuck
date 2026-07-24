"""Load and score the vendored BigCodeBench-Hard subset (v2 code study, Docker sandbox).

The BigCodeBench analogue of ``code_bench``: it turns the deterministic, sandbox-verified
subset (see ``scripts/vendor_bigcodebench.py``) into ``BenchmarkTask``s and supplies the
``score`` / ``feedback`` / in-loop ``exec_feedback`` the study injects. Unlike MBPP+ (which
saturated a capable worker — see the appendix), BigCodeBench-Hard is hard even for frontier
models, so GEPA has a real optimization gradient here.

Anti-oracle split (per task, precomputed at vendoring time): the suite's ``test_*`` methods
are partitioned into VISIBLE (run in-loop → ``bcb_feedback`` / ``exec_feedback``) and HIDDEN
(the held-out scorer → ``bcb_score``). The worker sees the function spec (``complete_prompt``)
and executable feedback from the visible methods, but never the hidden methods — so passing
what it can see does not guarantee the held-out score. Scoring executes untrusted model code
in the pinned Docker sandbox (``bcb_exec``), which enforces ``--network none`` and is
deterministic (fixed image + single-thread BLAS), so held-out scores replay bit-for-bit.
"""

from __future__ import annotations

import ast
import json
import random
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from whystuck.eval.bcb_exec import run_unittest
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.code_bench import extract_code  # shared markdown/code extraction
from whystuck.eval.code_exec import INFRA_VERDICTS, ExecResult

FeedbackLevel = Literal["lo", "mid", "hi"]

_DATA = Path(__file__).parent / "data" / "bigcodebench_hard.json"


def _payload(task: BenchmarkTask) -> dict[str, Any]:
    """The task's BigCodeBench payload, or a loud error if this isn't a bcb task."""
    payload = task.payload or {}
    for key in ("test", "visible_methods", "hidden_methods", "entry_point"):
        if key not in payload:
            raise KeyError(
                f"task {task.id!r} has no {key!r}; bcb_score/bcb_feedback need a task from "
                "load_bcb_tasks (wrong task-set / scorer wiring?)"
            )
    return payload


def _raise_if_infra(result: ExecResult, task: BenchmarkTask, suite: str) -> None:
    """Abort loudly on a sandbox/data fault instead of recording a fake 0.0 on a paid run.

    A ``sandbox_error`` (Docker/host fault) or ``no_assertions`` (a corrupt method split — the
    methods are trusted vendored data) means we never got a real verdict; folding it into 0.0
    would silently corrupt the study (a systematic fault would zero every arm and read as a
    null result). The dry-run preflight catches this before spend; this is the in-run backstop."""
    if result.verdict in INFRA_VERDICTS:
        raise RuntimeError(
            f"{suite} scoring for task {task.id!r} did not run (verdict={result.verdict}): "
            f"{result.detail}. Host/data fault, not a model failure — refusing to record 0.0. "
            "Check Docker + the whystuck-bcb image (run the bcb preflight) and re-run."
        )


def bcb_score(output: str, task: BenchmarkTask) -> float:
    """Held-out score: 1.0 iff the extracted solution passes the HIDDEN test methods.

    The worker never sees the hidden methods, so this cannot be gamed from inside the loop.
    Raises on an infrastructure verdict; a genuine model failure (wrong / timeout / early
    exit) scores 0.0."""
    payload = _payload(task)
    result = run_unittest(extract_code(output), payload["test"], payload["hidden_methods"])
    _raise_if_infra(result, task, "held-out (hidden)")
    return 1.0 if result.passed else 0.0


def _method_sources(test_src: str, names: Sequence[str]) -> str:
    """Source of the named ``test_*`` methods, concatenated — the 'hi' actionable payload.

    Only the VISIBLE method names are ever passed in, so this never exposes the hidden
    scorer. Returns "" on an unparseable suite (the caller then degrades 'hi' to 'mid')."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return ""
    want = set(names)
    segs = [
        seg
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in want
        if (seg := ast.get_source_segment(test_src, node))
    ]
    return "\n\n".join(segs)


def _shape_feedback(
    level: FeedbackLevel,
    *,
    passed: bool,
    n: int,
    detail: str,
    test_src: str,
    visible_methods: Sequence[str],
) -> str:
    """Shape the visible-test outcome at one of three actionability levels (code D2' lever).

    A pass reads the same at every level (the lever only shapes failures). On failure:
    ``lo`` = a bare verdict (no detail), ``mid`` = the raw failure detail (the pre-lever
    default, kept byte-identical), ``hi`` = the detail plus the source of the visible tests
    the solution must satisfy. Pure/deterministic (no execution), so it is unit-testable."""
    if passed:
        return f"All {n} visible test(s) pass (hidden edge-case tests may still fail)."
    if level == "lo":
        return "One or more visible tests failed; revise the solution."
    base = f"A visible test failed:\n{detail}"
    if level == "hi":
        src = _method_sources(test_src, visible_methods)
        if src:
            base += (
                "\n\nThe visible tests your solution must satisfy (fix the solution, not the "
                f"tests):\n```python\n{src}\n```"
            )
    return base


def bcb_feedback(output: str, task: BenchmarkTask, level: FeedbackLevel = "mid") -> str:
    """In-loop / reflection signal from the VISIBLE test methods only (never the hidden set).

    Used both as the loop's execution feedback (``exec_feedback``) and GEPA's reflection prose
    (``feedback_fn``). Raises on an infrastructure verdict rather than reporting a host fault
    as a test failure. ``level`` is the code-domain feedback-actionability lever (default
    ``mid`` = pre-lever behavior, byte-identical); see ``_shape_feedback``. Honest wording:
    passing the visible tests is necessary, not sufficient — hidden edge cases can still fail."""
    payload = _payload(task)
    result = run_unittest(extract_code(output), payload["test"], payload["visible_methods"])
    _raise_if_infra(result, task, "in-loop (visible)")
    return _shape_feedback(
        level,
        passed=result.passed,
        n=len(payload["visible_methods"]),
        detail=result.detail,
        test_src=payload["test"],
        visible_methods=payload["visible_methods"],
    )


def check_satisfiable(task: BenchmarkTask) -> bool:
    """True iff the stored reference passes BOTH the visible and hidden method subsets.

    Returns ``False`` for a wrong reference AND a host fault alike — use ``preflight_bcb_tasks``
    when you must distinguish them (e.g. before spending)."""
    payload = _payload(task)
    ref = payload.get("reference", "")
    return (
        bool(ref)
        and run_unittest(ref, payload["test"], payload["visible_methods"]).passed
        and run_unittest(ref, payload["test"], payload["hidden_methods"]).passed
    )


def preflight_bcb_tasks(tasks: list[BenchmarkTask]) -> None:
    """Pre-spend guard: prove Docker + the image + the data are sound on THIS host, or raise.

    Runs every task's REFERENCE through both method subsets in the sandbox, distinguishing a
    host/sandbox fault from corrupt data from a wrong reference, each with a precise message.
    Offline (``--network none``), no API key. Call it at the dry-run step so a broken sandbox
    (no Docker, missing image) or data file crashes BEFORE any spend, not as a silent all-zeros."""
    for task in tasks:
        payload = _payload(task)
        ref = payload.get("reference", "")
        if not ref:
            raise RuntimeError(f"preflight: task {task.id!r} has no reference solution to verify")
        for suite, methods in (
            ("visible", payload["visible_methods"]),
            ("hidden", payload["hidden_methods"]),
        ):
            result = run_unittest(ref, payload["test"], methods)
            if result.verdict in INFRA_VERDICTS:
                raise RuntimeError(
                    f"preflight: the Docker sandbox/data is broken on this host (task "
                    f"{task.id!r}, {suite} methods, verdict={result.verdict}): {result.detail}. "
                    "Fix Docker + the whystuck-bcb image before spending."
                )
            if not result.passed:
                raise RuntimeError(
                    f"preflight: the vendored REFERENCE for task {task.id!r} fails its own "
                    f"{suite} methods (verdict={result.verdict}): {result.detail}. Re-vendor "
                    "(scripts/vendor_bigcodebench.py)."
                )


def _render_goal(complete_prompt: str, entry_point: str) -> str:
    """The worker prompt: the BigCodeBench function spec + a request for the full function.

    We show ``complete_prompt`` (imports + signature + docstring, the standard BigCodeBench
    prompt) and ask for the complete function — the hidden tests are never shown."""
    return (
        f"Complete the following Python function. Return the COMPLETE function definition "
        f"named `{entry_point}`, including all necessary imports, inside a ```python code "
        f"block.\n\n{complete_prompt}"
    )


@lru_cache(maxsize=1)
def vendored_tasks() -> tuple[dict[str, Any], ...]:
    """The committed BigCodeBench-Hard subset as raw dicts (cached; read once)."""
    doc = json.loads(_DATA.read_text(encoding="utf-8"))
    return tuple(doc["tasks"])


def _to_benchmark_task(raw: dict[str, Any]) -> BenchmarkTask:
    return BenchmarkTask(
        id=str(raw["task_id"]).replace("/", "-"),
        payload={
            "goal": _render_goal(raw["complete_prompt"], raw["entry_point"]),
            "entry_point": raw["entry_point"],
            "test": raw["test"],
            "visible_methods": raw["visible_methods"],
            "hidden_methods": raw["hidden_methods"],
            "reference": raw["reference"],
        },
    )


Split = Literal["all", "train", "heldout"]


def load_bcb_tasks(
    split: Split = "all", *, n: int | None = None, seed: int = 0
) -> list[BenchmarkTask]:
    """Deterministically shuffle the vendored pool and split it into disjoint halves.

    Mirrors ``load_code_tasks``: same seed → same tasks; ``train``/``heldout`` are disjoint
    first/second halves; ``n`` caps the result and raises rather than silently truncating
    below the requested (and cost-priced) sample size."""
    tasks = [_to_benchmark_task(r) for r in vendored_tasks()]
    random.Random(seed).shuffle(tasks)
    if split != "all":
        mid = len(tasks) // 2
        tasks = tasks[:mid] if split == "train" else tasks[mid:]
    if n is not None:
        if n > len(tasks):
            raise ValueError(
                f"requested n={n} {split!r} bcb tasks but only {len(tasks)} available "
                f"(pool={len(vendored_tasks())}); lower n or vendor a larger pool — silent "
                "truncation would make the reported sample size disagree with what ran"
            )
        tasks = tasks[:n]
    return tasks
