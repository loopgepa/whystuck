"""Pins the paper's headline statistics to the committed results JSONs.

The article makes two quantitative claims that must not silently drift: (1) the
loop-level null Δ(D−B) is a tight, consistent zero across the fair tests, and
(2) the actionability effect Δ(hi−lo)=+0.063 has a 95% interval excluding zero.
These tests recompute both from the released per-seed results and pin them.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_deltas import (  # noqa: E402
    delta_arm,
    delta_between,
    exact_resample_ci,
    heldout_overlap,
    sign_test_p,
    task_set_of,
)

D = "docs"


def _resample_means(diffs: list[float]) -> list[float]:
    import itertools
    from statistics import mean

    return sorted(mean(s) for s in itertools.product(diffs, repeat=len(diffs)))


def test_neither_tail_of_the_interval_may_exceed_alpha_over_two():
    """The defining property of a percentile interval, checked on the exact distribution.

    The resample distribution is discrete with N = n**n equally-likely atoms, so at
    n = 3 a single atom carries 1/27 = 3.7% > 2.5% and therefore cannot be excluded
    from the interval. An implementation that drops it reports an interval that is too
    narrow on that side — and too narrow in the direction that turns an effect
    bracketing zero into one that appears to exclude it.
    """
    for diffs in ([-0.15, -0.15, 0.0], [0.090, 0.025, 0.075], [0.0, 0.055, 0.0, 0.0, -0.055]):
        _, lo, hi = exact_resample_ci(diffs)
        means = _resample_means(diffs)
        n = len(means)
        below = sum(1 for m in means if m < lo) / n
        above = sum(1 for m in means if m > hi) / n
        assert below <= 0.025 + 1e-12, f"lower tail {below:.4f} exceeds 2.5% for {diffs}"
        assert above <= 0.025 + 1e-12, f"upper tail {above:.4f} exceeds 2.5% for {diffs}"


def test_actuator_interval_includes_zero():
    """Regression pin: the actuator run's interval must NOT be reported as excluding zero.

    Per-seed Δ(D−B) = (−0.15, −0.15, 0.00). The all-zero resample (probability 1/27)
    is the largest atom, so the upper bound is exactly 0.0 and the interval includes it.
    """
    _, lo, hi = exact_resample_ci([-0.15, -0.15, 0.0])
    assert (lo, hi) == (-0.15, 0.0)
    assert lo < 0 <= hi  # includes zero: no "excludes zero" claim is available here


def test_sign_test_drops_ties_and_is_two_sided():
    # 3/3 same sign is the strongest possible at n=3: p = 2 * 0.5**3 = 0.25.
    assert sign_test_p([0.090, 0.025, 0.075]) == 0.25
    # A tie is dropped (standard sign test), leaving n=2: p = 2 * 0.5**2 = 0.50.
    assert sign_test_p([-0.15, -0.15, 0.0]) == 0.5
    # All ties: no signed observations, so no evidence against the null.
    assert sign_test_p([0.0, 0.0, 0.0]) == 1.0
    # Mixed signs are weaker than unanimous ones.
    assert sign_test_p([0.05, -0.05, 0.05]) == 1.0


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


# --- seed independence -------------------------------------------------------
#
# Both statistics above resample PER-SEED differences, so both assume seeds are
# independent replications. For the vendored task sets they are not: the loader shuffles
# a fixed file, so a seed only picks a partition and two seeds share ~half their held-out
# tasks. Correlated units make the interval too narrow and the sign-test p too small,
# both toward overstating significance, so the analysis has to know which is which.


def test_task_set_is_resolved_for_every_released_run():
    import glob
    import json

    for path in sorted(glob.glob(f"{D}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)["meta"]
        assert task_set_of(meta) in {"ifbench", "multi", "multi-hard", "code", "bcb"}, path


def test_task_set_prefers_the_explicit_key_over_the_dataset_string():
    assert task_set_of({"task_set": "bcb", "dataset": "IFBench"}) == "bcb"


def test_task_set_falls_back_to_the_dataset_string_for_older_runs():
    # The earliest runs predate meta["task_set"]; `dataset` is present in all of them.
    assert task_set_of({"dataset": "IFBench"}) == "ifbench"
    assert task_set_of({"dataset": "multi-constraint-hard (IFEval-style)"}) == "multi-hard"
    assert task_set_of({"dataset": "multi-constraint (IFEval-style)"}) == "multi"


def test_task_set_raises_on_an_unrecognized_dataset():
    # Guessing would silently grant replication to a run that may not have it.
    with pytest.raises(ValueError, match="cannot determine"):
        task_set_of({"dataset": "some new benchmark"})


def test_measured_overlap_matches_the_documented_split_behaviour():
    # The claim the correction rests on, measured rather than asserted.
    assert heldout_overlap("multi-hard", [0, 1, 2]) == 0.0
    for ts in ("ifbench", "code", "bcb"):
        frac = heldout_overlap(ts, [0, 1, 2])
        assert 0.35 < frac < 0.65, (ts, frac)
