"""Pins the paper's headline statistics to the committed results JSONs.

The article makes two quantitative claims that must not silently drift: (1) the
loop-level null Δ(D−B) is a tight, consistent zero across the fair tests, and
(2) the actionability effect Δ(hi−lo)=+0.063 has a 95% interval excluding zero.
These tests recompute both from the released per-seed results and pin them.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_deltas import delta_arm, delta_between, exact_resample_ci  # noqa: E402

D = "docs"


def test_delta_D_minus_B_is_exactly_zero_where_arms_converge():
    # runs 4 and 10: B and D optimize to byte-identical genomes every seed → exact [0,0].
    for path in (f"{D}/validation-study-feedback.json", f"{D}/validation-study-fbacthi.json"):
        diffs = delta_arm(path)
        assert diffs == [0.0, 0.0, 0.0]
        m, lo, hi = exact_resample_ci(diffs)
        assert (m, lo, hi) == (0.0, 0.0, 0.0)


def test_delta_D_minus_B_null_is_tight_and_brackets_zero_on_run3():
    # run 3 (n=5): per-seed {0,+0.055,0,0,-0.055}; the interval is tight and includes 0.
    diffs = delta_arm(f"{D}/validation-study-multihard.json")
    m, lo, hi = exact_resample_ci(diffs)
    assert abs(m) < 1e-9  # nets to zero
    assert lo < 0 < hi  # brackets zero
    assert hi - lo < 0.08  # tight (|bounds| ≈ 0.033), not an underpowered wide band


def test_delta_hi_minus_lo_is_positive_and_excludes_zero():
    # the one supra-threshold effect: actionable feedback beats a bare verdict.
    diffs = delta_between(
        f"{D}/validation-study-fbacthi.json", f"{D}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(diffs)
    assert m > 0.05  # clears the pre-registered +0.05 threshold (≈ +0.063)
    assert lo > 0  # 95% interval excludes zero even at n=3
    assert all(x > 0 for x in diffs)  # every seed positive


def test_exact_resample_ci_is_deterministic_and_ordered():
    # No RNG: identical inputs → identical interval, and lo ≤ mean ≤ hi.
    d = [0.0, 0.055, 0.0, 0.0, -0.055]
    a = exact_resample_ci(d)
    b = exact_resample_ci(d)
    assert a == b
    m, lo, hi = a
    assert lo <= m <= hi
