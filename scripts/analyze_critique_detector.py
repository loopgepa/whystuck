#!/usr/bin/env python
"""Measure the in-loop critique as a failure detector (zero cost, offline).

The loop-level GEPA null (see docs/validation-study-summary.md) has a mechanistic
explanation: the intrinsic self-critique is a near-random detector of whether a
draft satisfies its constraints, so evolving its rubric cannot systematically
help. This script measures that directly. It reuses the PRODUCTION self-refine
loop on the seed genome and REPLAYS each run's recorded cassette (no network, no
spend), pairing every critique's PASS/FAIL verdict with the ground-truth
`mc_pass` of the exact draft it judged, then reports the confusion matrix and F1.

    python scripts/analyze_critique_detector.py
"""

from __future__ import annotations

import math

from whystuck.eval.cassette import Cassette, CassetteMiss
from whystuck.eval.llm_worker import make_self_refine_run_task
from whystuck.eval.multi_constraint import load_multi_constraint, mc_feedback, mc_pass, mc_score
from whystuck.eval.study import seed_genome
from whystuck.eval.text_stats import text_stats

CASSETTE_DIR = "tests/fixtures/cassettes"
_N_HELDOUT = 10


def collect_events(
    cassette_path: str,
    seeds: list[int],
    *,
    loop_kw: dict,
    n_heldout: int = _N_HELDOUT,
    gene_overrides: dict[str, str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[bool, bool]]:
    """Return (verdict_pass, actual_pass) for every first-critique event.

    `verdict_pass` is True when the critique emitted PASS ("draft is good");
    `actual_pass` is the ground-truth mc_pass of the draft the critique judged.
    Seeds whose calls are absent from the cassette raise on replay and are
    skipped, so passing a superset of seeds is safe.

    `n_heldout` and `gene_overrides` must match the run being replayed or every call
    misses: the generated pool size is derived from `n_heldout` exactly as
    `run_validation_study` derives it, and an overridden gene changes the critique prompt
    and therefore its cassette key. `gene_overrides` is what lets a hand-written-gene
    ablation (`--seed-gene`) be measured as a detector alongside the runs it is compared
    with. `exclude` must match it for the same reason: it is the p₀ lever, it changes the
    generated pool, and replaying an excluded run without it misses every call.

    Note what `actual_pass` measures on the FIRST critique event: the ground truth of the
    draft *before* any refinement. Its mean over first events is therefore exactly p₀, the
    base rate of already-correct first drafts --- the axis the p₀ sweep varies. So p₀ is
    measurable from any recorded cassette rather than having to be estimated, which matters
    because the independence estimate is biased: constraint failures correlate, and on
    multi-hard it overestimates the measured value by roughly a factor of two."""
    include_plan = "plan_gene" in loop_kw
    # Same rule run_validation_study uses, derived rather than duplicated as a constant so
    # the two cannot drift apart and silently generate a different task pool.
    pool = max(64, 2 * n_heldout + 2)
    events: list[tuple[bool, bool]] = []
    skipped = 0
    with Cassette(cassette_path, mode="replay") as cas:

        def _unreachable(_prompt: str) -> str:
            # Never invoked in replay: Cassette.wrap raises CassetteMiss on a miss
            # *before* calling this, and returns the cached value on a hit.
            raise AssertionError("unreachable in replay mode")

        run_task = make_self_refine_run_task(
            cas.wrap(_unreachable, name="worker"),
            mc_score,
            feedback=mc_feedback,
            max_retries=1,
            **loop_kw,
        )
        for seed in seeds:
            # load_multi_constraint raises loudly on a pool/seed misconfig — do NOT
            # catch it: a silently dropped seed would shrink n and misstate the sample.
            heldout = load_multi_constraint(
                "heldout", seed=seed, n=n_heldout, pool=pool, hard=True, exclude=exclude
            )
            genome = seed_genome(
                "study-A",
                evolvable=set(),
                include_plan=include_plan,
                overrides=gene_overrides,
            )
            for task in heldout:
                try:
                    trace = run_task(genome, task)
                except CassetteMiss:
                    # Only a genuinely un-recorded call is skippable (passing a superset
                    # of seeds is intended); a real code regression must still propagate.
                    skipped += 1
                    continue
                gen0 = next((s for s in trace.steps if s.step_id == "generate-0"), None)
                crit0 = next((s for s in trace.steps if s.step_id == "critique-0"), None)
                if gen0 is None or crit0 is None:
                    continue
                verdict_pass = (crit0.output_summary or "").strip().upper().startswith("PASS")
                actual_pass = mc_pass(gen0.output_summary or "", task)
                events.append((verdict_pass, actual_pass))
    if skipped:
        print(f"  (skipped {skipped} un-recorded task evals in {cassette_path})")
    return events


def detector_stats(events: list[tuple[bool, bool]]) -> dict[str, float]:
    """Confusion matrix + precision/recall/F1 for PASS ("draft is good")."""
    tp = sum(1 for v, a in events if v and a)
    fp = sum(1 for v, a in events if v and not a)
    fn = sum(1 for v, a in events if not v and a)
    tn = sum(1 for v, a in events if not v and not a)
    n = len(events)
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not math.isnan(prec) else math.nan
    return {
        "n": n,
        "base_rate": (tp + fn) / n if n else math.nan,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else math.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def base_rate_p0(events: list[tuple[bool, bool]]) -> float:
    """p₀: the share of first drafts that already satisfied every constraint.

    This is the quantity the p₀ sweep varies and the quantity the always-FIX mechanism is a
    function of: an unconditional-refine critique can only lose where a draft was already
    correct, so p₀ bounds what that policy costs on the training signal. Measured, not
    estimated -- see `collect_events` on why the independence estimate runs ~2x high.
    """
    if not events:
        raise ValueError(
            "base_rate_p0 called with no events; returning 0.0 would report 'no draft was "
            "ever correct', which is a finding, not an empty measurement"
        )
    return sum(1 for _, actual in events if actual) / len(events)


_RUNS = [
    ("run3 hard (intrinsic)", "study-multihard.json", [0, 1, 2, 3, 4], {}),
    ("run4 feedback (tool-fed)", "study-feedback.json", [0, 1, 2], {"stats_fn": text_stats}),
    ("run5 plan (intrinsic)", "study-plan.json", [0, 1, 2], {"plan_gene": "plan"}),
]


def main() -> int:  # pragma: no cover - CLI convenience
    for tag, name, seeds, kw in _RUNS:
        ev = collect_events(f"{CASSETTE_DIR}/{name}", seeds, loop_kw=kw)
        s = detector_stats(ev)
        print(
            f"{tag}: n={s['n']} base={s['base_rate']:.2f} prec={s['precision']:.2f} "
            f"rec={s['recall']:.2f} f1={s['f1']:.2f} acc={s['accuracy']:.2f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
