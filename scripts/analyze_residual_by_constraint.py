#!/usr/bin/env python
"""Decompose the fix-step residual BY CONSTRAINT (zero cost, offline).

Runs 7-10 show the self-refine loop's held-out ceiling is set by the *fix* step,
not detection, and that even a perfect detector + maximally-actionable feedback
saturates below 1.0 (a generator-limited residual). This script asks *where* that
residual lives: it replays each run's recorded cassette (no network, no spend),
takes the FINAL draft the loop produced for every held-out task, and reports the
per-constraint satisfaction rate. The finding: the residual is almost entirely one
constraint — `exact_word_count` (a known LLM skill gap) — while every "choice"
constraint is essentially solved once feedback is actionable.

    python scripts/analyze_residual_by_constraint.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from whystuck.eval.cassette import Cassette, CassetteMiss
from whystuck.eval.llm_worker import make_self_refine_run_task
from whystuck.eval.multi_constraint import (
    MC_VERIFIERS,
    _pairs,
    load_multi_constraint,
    mc_feedback,
    mc_score,
)
from whystuck.eval.study import seed_genome

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_validation_study import _build_oracle_feedback  # noqa: E402

CASSETTE_DIR = "tests/fixtures/cassettes"
_N_HELDOUT = 10
_POOL = max(64, 2 * _N_HELDOUT + 2)


def _final_draft(trace) -> str:
    """The last worker (generate/refine) output — the draft the loop actually scored."""
    workers = [s for s in trace.steps if s.role == "worker"]
    return (workers[-1].output_summary or "") if workers else ""


def constraint_pass_counts(
    cassette_path: str,
    seeds: list[int],
    *,
    oracle_critique: bool = False,
    feedback_level: str = "mid",
    max_retries: int = 1,
) -> dict[str, list[int]]:
    """Replay the seed loop and tally [passed, total] per constraint id on FINAL drafts.

    Uses the PRODUCTION self-refine loop in strict replay mode, with the exact
    `oracle_feedback` the paid run used (so the replayed prompts hit the cassette).
    Seeds whose calls are absent raise on replay and are skipped, so passing a
    superset of seeds is safe.

    MAINTAINER NOTE: the CassetteMiss/skip guard catches a wrong `oracle_critique`
    or `feedback_level` (those change the prompt text -> unseen hash), but NOT a
    `max_retries` set *lower* than the recording: a shortened loop terminates early
    on a prefix of the recorded calls (all hits) and would silently score an
    earlier, unrefined draft as "final". So `_REGIMES` must keep max_retries in
    lock-step with `verify_reproducibility.py`'s RUNS; re-check both together."""
    oracle_feedback = (
        _build_oracle_feedback(feedback_level, mc_feedback) if oracle_critique else None
    )
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    skipped = 0
    with Cassette(cassette_path, mode="replay") as cas:

        def _unreachable(_prompt: str) -> str:  # pragma: no cover - never hit in replay
            raise AssertionError("live model call during replay")

        run_task = make_self_refine_run_task(
            cas.wrap(_unreachable, name="worker"),
            mc_score,
            feedback=mc_feedback,
            max_retries=max_retries,
            oracle_critique=oracle_critique,
            oracle_feedback=oracle_feedback,
        )
        for seed in seeds:
            heldout = load_multi_constraint(
                "heldout", seed=seed, n=_N_HELDOUT, pool=_POOL, hard=True
            )
            genome = seed_genome("study-A", evolvable=set())
            for task in heldout:
                try:
                    trace = run_task(genome, task)
                except CassetteMiss:
                    skipped += 1
                    continue
                final = _final_draft(trace)
                for iid, kw in _pairs(task):
                    counts[iid][1] += 1
                    counts[iid][0] += 1 if MC_VERIFIERS[iid](final, kw) else 0
    if skipped:
        print(f"  (skipped {skipped} un-recorded task evals in {cassette_path})")
    return dict(counts)


def pass_rate(counts: dict[str, list[int]], iid: str) -> float:
    passed, total = counts.get(iid, [0, 0])
    return passed / total if total else float("nan")


# (tag, cassette, seeds, oracle_critique, feedback_level, max_retries)
_REGIMES = [
    ("run3 intrinsic R1", "study-multihard.json", [0, 1, 2, 3, 4], False, "mid", 1),
    ("run8 oracle mid R3", "study-fixbudget.json", [0, 1, 2], True, "mid", 3),
    ("run10 oracle hi R3", "study-fbacthi.json", [0, 1, 2], True, "hi", 3),
]
_IDS = [
    "exact_word_count",
    "word_max_length",
    "forbidden_word",
    "num_bullets",
    "end_with",
    "include_keyword",
    "all_lowercase",
    "no_comma",
]


def main() -> int:  # pragma: no cover - CLI convenience
    tallies = {
        tag: constraint_pass_counts(
            f"{CASSETTE_DIR}/{name}",
            seeds,
            oracle_critique=oc,
            feedback_level=lvl,
            max_retries=r,
        )
        for tag, name, seeds, oc, lvl, r in _REGIMES
    }
    header = "  ".join(f"{tag.split()[0]:>7}" for tag, *_ in _REGIMES)
    print(f"{'constraint':18} {header}")
    for iid in _IDS:
        cells = []
        for tag, *_ in _REGIMES:
            _, total = tallies[tag].get(iid, [0, 0])
            cells.append(f"{pass_rate(tallies[tag], iid):.2f}({total})" if total else "  -  ")
        print(f"{iid:18} " + "  ".join(f"{c:>7}" for c in cells))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
