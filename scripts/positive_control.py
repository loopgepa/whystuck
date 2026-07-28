#!/usr/bin/env python
"""Positive control: can the corrected harness detect a loop-level advantage that exists?

Every headline in this study is a null, and a null is only worth reading if the instrument
that produced it can register a signal. This script builds a landscape where evolving the
verify gene is the *only* path to a passing answer, runs the real four-arm study through
real GEPA (no stubbed optimizer), and checks that arm D beats arm B by a wide margin.

The landscape, by construction:

  * the generator always drafts ``x x``, which fails the task's single constraint
    (``format:no_whitespace``) no matter how its prompt is worded;
  * the critique catches the flaw only when the verify gene contains the sentinel, and a
    caught flaw triggers a refine to a passing answer;
  * so arm B (evolves ``generate`` only) cannot escape the failing draft however well it
    optimizes, while arm D (also evolves ``ensure_correct``) can.

No network and no spend: the worker and reflection LMs are local functions. The reflection
LM returns a GEPA-formatted proposal that inserts the sentinel, which is what lets real GEPA
run its accept/reject machinery over a candidate that genuinely improves the training score.

    python scripts/positive_control.py     # (needs the eval venv: gepa must be importable)

Exit 0 iff arm D's held-out beats arm B's by at least MIN_GAP. Prints the four arm scores so
a reader can see the instrument responding, not just a pass/fail.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

MIN_GAP = 0.5
SENTINEL = "CHECK-WHITESPACE"


def _worker(prompt: str, **_kw: object) -> str:
    """generate -> ``x x`` (fails); critique -> catches only with the sentinel; refine -> passes."""
    if "# Draft answer" in prompt:  # critique step
        return "FAIL: the draft contains whitespace; remove it" if SENTINEL in prompt else "PASS"
    if "# Problems to fix" in prompt:  # refine step
        return "solid"
    return "x x"


def _reflection(prompt: str, **_kw: object) -> str:
    """A proposal that inserts the sentinel, in the fenced form GEPA parses."""
    return f"```\n{SENTINEL}: before answering, verify the draft has no whitespace.\n```"


def main() -> int:  # pragma: no cover - CLI
    try:
        import gepa  # noqa: F401
    except ImportError:
        print("gepa is not importable; run this with the eval venv (scratchpad/evalvenv).")
        return 2

    from whystuck.core.storage import WhyStuckStore
    from whystuck.eval.benchmark import BenchmarkTask
    from whystuck.eval.study import run_study

    tasks = [
        BenchmarkTask(
            id=f"pc{i}",
            payload={"goal": f"answer prompt {i}", "instruction_id_list": ["format:no_whitespace"]},
        )
        for i in range(4)
    ]

    with tempfile.TemporaryDirectory() as d:
        store = WhyStuckStore(Path(d))
        results = run_study(
            store,
            complete=_worker,
            reflection_lm=_reflection,
            train=tasks,
            heldout=tasks,
            arms=("A", "B", "C", "D"),
            max_metric_calls=24,
            loop_max_retries=1,
        )

    print("=== positive control: a landscape where the verify gene is the only lever ===")
    for arm in "ABCD":
        r = results[arm]
        print(f"  {arm}: seed {r.seed_heldout:.3f} -> optimized {r.opt_heldout:.3f}")
    gap = results["D"].opt_heldout - results["B"].opt_heldout
    print(f"\nΔ(D−B) = {gap:+.3f}   (need ≥ {MIN_GAP} for the instrument to count as responsive)")
    if gap >= MIN_GAP:
        print("PASS — the harness detects a loop-level advantage when one exists.")
        return 0
    print("FAIL — the harness did NOT register an advantage that is true by construction.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
