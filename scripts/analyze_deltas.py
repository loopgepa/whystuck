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
    means. Exact at the small n used here (5**5 = 3125, 3**3 = 27)."""
    if not diffs:
        raise ValueError("no differences given")
    n = len(diffs)
    means = sorted(mean(s) for s in itertools.product(diffs, repeat=n))
    lo = means[int((alpha / 2) * len(means))]
    hi = means[int((1 - alpha / 2) * len(means)) - 1]
    return mean(diffs), lo, hi


# (label, per-seed differences) for the fair runs' D−B null.
_NULL_RUNS = [
    ("run3 hard-multi (n=5)", f"{DOCS}/validation-study-multihard.json"),
    ("run4 feedback  (n=3)", f"{DOCS}/validation-study-feedback.json"),
    ("run10 oracle-hi(n=3)", f"{DOCS}/validation-study-fbacthi.json"),
]


def main() -> int:  # pragma: no cover - CLI convenience
    print("=== null: Δ(D−B) per-seed, mean, exact-resample 95% CI ===")
    for label, path in _NULL_RUNS:
        d = delta_arm(path)
        m, lo, hi = exact_resample_ci(d)
        seeds = [f"{x:+.3f}" for x in d]
        print(f"{label}: Δ={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]")
    print("\n=== effect: Δ(hi−lo) actionability, paired by seed (n=3) ===")
    d = delta_between(
        f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(d)
    excl = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
    seeds = [f"{x:+.3f}" for x in d]
    print(f"Δ(hi−lo)={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]  ({excl})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
