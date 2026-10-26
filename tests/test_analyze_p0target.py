"""The pre-registered statistics of the target-p0 study (2026-10-25 pre-registration).

The primary quantity is the swing S = mean(Δ_mid) − mean(Δ_low) over independent seeds. Its
interval is a percentile bootstrap resampling seeds independently within each level, and its
test is Welch's t. These are pinned against hand-computable cases so the verdict cannot hinge
on an off-by-one in a quantile, which is how this study once published an interval as
excluding zero when it did not.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_p0target.py"


def _mod():
    spec = importlib.util.spec_from_file_location("analyze_p0target", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_swing_point_estimate_and_bootstrap_contains_it():
    m = _mod()
    lo = [0.0, 0.1, -0.1, 0.0, 0.05, -0.05, 0.0, 0.0, 0.1, -0.1]
    mid = [x + 0.2 for x in lo]
    s = m.swing(mid, lo, n_boot=2000, seed=0)
    assert s["S"] == pytest.approx(0.2)
    assert s["ci"][0] < 0.2 < s["ci"][1]
    assert s["ci"][0] > 0  # a clean, consistent shift excludes zero


def test_swing_bootstrap_is_deterministic_and_registered_size():
    m = _mod()
    a, b = [0.1, 0.0, 0.2], [0.0, 0.1, -0.1]
    assert m.swing(a, b, seed=0) == m.swing(a, b, seed=0)
    assert m.N_BOOT == 10_000


def test_swing_of_identical_levels_is_centered_and_straddles_zero():
    m = _mod()
    x = [0.05, -0.05, 0.1, -0.1, 0.0, 0.02, -0.02, 0.03, -0.03, 0.0]
    s = m.swing(x, list(x), n_boot=4000, seed=1)
    assert s["S"] == pytest.approx(0.0)
    assert s["ci"][0] < 0 < s["ci"][1]


def test_welch_matches_a_hand_computation():
    m = _mod()
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0]
    t, df = m.welch(a, b)
    # means 2.5, 4.0; sample vars 1.6667, 4.0; se^2 = 1.6667/4 + 4/3 = 1.75
    assert t == pytest.approx((2.5 - 4.0) / math.sqrt(1.75))
    num = 1.75**2
    den = (1.6667 / 4) ** 2 / 3 + (4.0 / 3) ** 2 / 2
    assert df == pytest.approx(num / den, rel=1e-3)


def test_verdict_rule():
    m = _mod()
    assert m.verdict(0.1, (0.02, 0.18)) == "SUPPORTED"
    assert m.verdict(-0.1, (-0.18, -0.02)) == "CONTRADICTED"
    assert m.verdict(0.05, (-0.01, 0.11)) == "INCONCLUSIVE"


def test_decomposition_is_an_identity():
    m = _mod()
    d = m.decompose(draft_b=0.6, final_b=0.8, draft_d=0.55, final_d=0.85)
    assert d["delta"] == pytest.approx(0.05)
    assert d["draft"] + d["loop"] == pytest.approx(d["delta"])
    assert d["draft"] == pytest.approx(-0.05)
    assert d["loop"] == pytest.approx(0.10)


def test_pooled_bootstrap_is_the_seed_resampled_mean():
    m = _mod()
    x = [0.0, 0.1, -0.1, 0.05, -0.05, 0.0]
    p = m.pooled(x, n_boot=4000, seed=0)
    assert p["mean"] == pytest.approx(0.0)
    assert p["n"] == 6
    assert p["ci"][0] < 0 < p["ci"][1]
    assert m.pooled(x, n_boot=4000, seed=0) == p
