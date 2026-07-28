#!/usr/bin/env python
"""Compute the study's headline effect sizes + intervals from the committed results.

Zero cost, offline: reads the released per-seed results JSONs (no cassette replay,
no model) and derives the two load-bearing statistics the paper reports, so the
article's numbers are backed by a reproducible computation rather than hand math:

  1. the null  Δ(D−B) = mean_seed(D.opt_heldout − B.opt_heldout), per fair run;
  2. the one positive effect Δ(hi−lo) = mean_seed(hi.A.seed_heldout − lo...),

each with an EXACT-resample 95% interval (all n**n resamples of the paired
per-seed differences — exact at the small n here, no RNG, fully deterministic).

    python scripts/analyze_deltas.py
"""

from __future__ import annotations

import itertools
import json
import math
from statistics import mean

DOCS = "docs"


def _runs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def delta_arm(
    results_json: str, a: str = "D", b: str = "B", field: str = "opt_heldout"
) -> list[float]:
    """Per-seed paired difference arm_a − arm_b on `field` (default D−B on opt held-out)."""
    return [r[a][field] - r[b][field] for r in _runs(results_json)]


def delta_between(
    json_hi: str, json_lo: str, arm: str = "A", field: str = "seed_heldout"
) -> list[float]:
    """Per-seed paired difference between two runs (same seeds/tasks), e.g. hi−lo on the
    seed held-out. Requires equal seed counts (paired by seed index)."""
    hi, lo = _runs(json_hi), _runs(json_lo)
    if len(hi) != len(lo):
        raise ValueError(f"seed-count mismatch: {json_hi} has {len(hi)}, {json_lo} has {len(lo)}")
    return [h[arm][field] - lo[i][arm][field] for i, h in enumerate(hi)]


def exact_resample_ci(diffs: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) with an EXACT percentile bootstrap: enumerate all n**n resamples
    (deterministic, no RNG) and take the alpha/2 and 1−alpha/2 percentile of the resample
    means. Exact at the small n used here (5**5 = 3125, 3**3 = 27).

    Indexing is `ceil(q * N) - 1`, the smallest index k with P(X <= means[k]) >= q. This
    matters because the resample distribution is DISCRETE: at n = 3 each of the 27 atoms
    carries 1/27 = 3.70% > alpha/2, so no single atom may be dropped from either tail.
    A previous implementation used `int(q * N) - 1` for the upper bound, which discarded
    the largest atom and reported an interval too narrow on that side — narrow in exactly
    the direction that makes an effect bracketing zero look like it excludes zero. The
    invariant the tests pin is per-tail: P(X < lo) <= alpha/2 and P(X > hi) <= alpha/2.
    """
    if not diffs:
        raise ValueError("no differences given")
    n = len(diffs)
    means = sorted(mean(s) for s in itertools.product(diffs, repeat=n))
    total = len(means)
    lo = means[math.ceil((alpha / 2) * total) - 1]
    hi = means[math.ceil((1 - alpha / 2) * total) - 1]
    return mean(diffs), lo, hi


def sign_test_p(diffs: list[float]) -> float:
    """Two-sided exact sign-test p-value on the paired per-seed differences.

    Ties (exactly 0.0) are dropped, per the standard sign test: a seed on which the two
    arms optimized to byte-identical genomes contributes no signed information — the
    cassette serves both arms the same completion, so the 0.0 is an identity, not an
    observation. Reported alongside the interval because at these seed counts the sign
    test is the honest floor: n = 3 unanimous gives p = 0.25 at best, so no result here
    can reach conventional significance on sign alone, and any "excludes zero" claim
    rests on the magnitudes rather than on replication across seeds.
    """
    signed = [d for d in diffs if d != 0.0]
    n = len(signed)
    if n == 0:
        return 1.0
    k = sum(1 for d in signed if d > 0)
    # Two-sided: double the smaller tail of Binomial(n, 0.5), clamped at 1.0.
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    sf = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * min(cdf, sf))


# (label, per-seed differences) for the fair runs' D−B null. Includes the cross-domain
# BigCodeBench-Hard run (D5); MBPP+ is excluded because it saturated its training split
# (all arms == seed, a degenerate zero — see the paper appendix), not a fair test.
#
# Run 10 (fbacthi) was previously listed here and has been REMOVED: it sets
# `oracle_critique`, under which the verify gene is inert by construction
# (`llm_worker.py`: "Not a D-vs-B test — the verify gene cannot matter under a perfect
# oracle"). Its Δ(D−B) = 0 is an identity, not a measurement, so citing its [0,0] as
# evidence that the null is tight was circular. The same applies to any oracle run.
_NULL_RUNS = [
    ("run3  hard-multi    (n=5)", f"{DOCS}/validation-study-multihard.json"),
    ("run4  feedback      (n=3)", f"{DOCS}/validation-study-feedback.json"),
    ("D5    bcb-hard code (n=2)", f"{DOCS}/validation-study-bcb.json"),
    ("run11 nano worker   (n=3)", f"{DOCS}/validation-study-multihard-nano.json"),
]


def _describe(label: str, d: list[float]) -> str:
    """One line per run: point estimate, interval, sign test, and degenerate-seed count.

    `ties` is the load-bearing caveat. A tie means B and D optimized to byte-identical
    genomes, so the cassette served both arms the same completion and the 0.0 is forced.
    Reporting it separates "the treatment was applied and did nothing" from "the treatment
    was never applied", which a bare interval of [0,0] silently conflates.
    """
    m, lo, hi = exact_resample_ci(d)
    ties = sum(1 for x in d if x == 0.0)
    seeds = [f"{x:+.3f}" for x in d]
    return (
        f"{label}: Δ={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]"
        f"  sign p={sign_test_p(d):.2f}  ties={ties}/{len(d)}"
    )


def main() -> int:  # pragma: no cover - CLI convenience
    print("=== null: Δ(D−B) per-seed, mean, exact-resample 95% CI, sign test, ties ===")
    print("(ties = seeds where B and D produced byte-identical genomes: forced zeros)")
    for label, path in _NULL_RUNS:
        print(_describe(label, delta_arm(path)))
    print("\n=== effect: Δ(hi−lo) actionability, paired by seed (n=3) ===")
    print("NOTE: measured under --oracle-critique, which gates on the TRUE metric at")
    print("held-out scoring time too. This is a ceiling result: it says richer repair")
    print("feedback helps GIVEN a perfect checker, not that it helps in deployment.")
    d = delta_between(
        f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(d)
    excl = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
    seeds = [f"{x:+.3f}" for x in d]
    print(
        f"Δ(hi−lo)={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]  ({excl})"
        f"  sign p={sign_test_p(d):.2f}"
    )

    dc = delta_between(
        f"{DOCS}/validation-study-bcb-acthi.json", f"{DOCS}/validation-study-bcb-actlo.json"
    )
    print(f"\n=== code actionability: Δ(hi−lo) arm-A held-out, BigCodeBench-Hard (n={len(dc)}) ===")
    mc, loc, hic = exact_resample_ci(dc)
    exclc = "excludes 0" if loc > 0 or hic < 0 else "includes 0 (suggestive, not significant)"
    seedsc = [f"{x:+.3f}" for x in dc]
    print(f"Δ(hi−lo)={mc:+.3f}  seeds={seedsc}  95%CI[{loc:+.3f},{hic:+.3f}]  ({exclc})")

    print("\n=== actuator experiment: Δ(D−B) aiming GEPA at the fixer's feedback gene ===")
    for label, path in (
        ("actuator small (n=2, 8/10, mmc30)", f"{DOCS}/validation-study-bcb-actuator.json"),
        ("actuator powered (n=3, 20/20, mmc50)", f"{DOCS}/validation-study-bcb-actuator-big.json"),
    ):
        da = delta_arm(path)
        _, loa, hia = exact_resample_ci(da)
        # A negative effect "excludes zero" only when its upper bound < 0. With the
        # percentile index corrected, the powered run's upper bound is exactly 0.0 (the
        # all-zero resample is the largest atom and carries 1/27 > alpha/2), so this run
        # does NOT exclude zero — the earlier claim that it did was an artifact of
        # dropping that atom. A degenerate [0,0] interval is reported as "point zero".
        excla = "excludes zero" if hia < 0 else ("point zero" if loa == hia == 0 else "includes 0")
        print(f"{_describe(label, da)}  ({excla})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
