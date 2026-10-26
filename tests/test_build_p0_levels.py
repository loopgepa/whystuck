"""The registered baseline rule of the target-p0 study (deviation 3), and its fallback order."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_p0_levels.py"


def _mod():
    spec = importlib.util.spec_from_file_location("build_p0_levels", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_b_star_is_the_mean_of_pool_means_and_is_tried_first():
    order = _mod().registered_baselines([0.72, 0.94])
    assert order[0] == pytest.approx(0.83)


def test_fallback_stays_within_five_hundredths_nearest_first():
    order = _mod().registered_baselines([0.80, 0.80])
    assert len(order) == 11
    assert max(abs(b - 0.80) for b in order) == pytest.approx(0.05)
    dists = [round(abs(b - 0.80), 6) for b in order]
    assert dists == sorted(dists)


def test_the_registered_design_constants():
    m = _mod()
    assert m.TARGETS == {"low": 0.07, "mid": 0.27, "high": 0.50}
    assert list(m.TARGETS) == ["low", "mid", "high"]  # fill order, deviation 3
    assert (m.N_TRAIN, m.N_HELDOUT, len(m.SEEDS)) == (8, 10, 10)
    assert m.TOL_P0 == m.TOL_BASELINE == 0.03
    assert m.SHARE == (0.5, 0.07)
