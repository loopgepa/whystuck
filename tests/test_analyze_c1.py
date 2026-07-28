"""Tests for the PRE-REGISTERED C1 analysis (scripts/analyze_c1.py).

Written alongside the script and committed before the run was paid for, so the decision
rule could not be tuned to the result. Every case below uses synthetic differences.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_c1 import (  # noqa: E402
    SESOI,
    bootstrap_ci,
    paired_by_task,
    verdict,
)


def _write(path, *, per_task, second=None):
    """A minimal results JSON with one seed and one arm A carrying per-task scores."""
    agg = (sum(per_task.values()) / len(per_task)) if per_task else 0.0
    arm = {
        "name": "A",
        "seed_heldout": agg,
        "opt_heldout": agg,
        "seed_heldout_by_task": per_task,
        "opt_heldout_by_task": second if second is not None else per_task,
    }
    path.write_text(json.dumps({"meta": {}, "runs": [{"A": arm}]}), encoding="utf-8")
    return str(path)


def test_pairs_by_task_id(tmp_path):
    base = _write(tmp_path / "b.json", per_task={"t1": 0.5, "t2": 0.25})
    hand = _write(tmp_path / "h.json", per_task={"t2": 0.75, "t1": 0.5})  # order differs
    assert paired_by_task(base, hand) == {"t1": 0.0, "t2": 0.5}


def test_refuses_mismatched_task_sets(tmp_path):
    # A silent intersection would quietly change the sample the test is run on.
    base = _write(tmp_path / "b.json", per_task={"t1": 0.5, "t2": 0.5})
    hand = _write(tmp_path / "h.json", per_task={"t1": 0.5, "t3": 0.5})
    with pytest.raises(ValueError, match="different task sets"):
        paired_by_task(base, hand)


def test_refuses_a_nondeterministic_condition(tmp_path):
    # Arm A scores held-out twice; under a content-addressed cassette the passes must agree.
    base = _write(
        tmp_path / "b.json", per_task={"t1": 0.5, "t2": 0.5}, second={"t1": 0.5, "t2": 0.25}
    )
    hand = _write(tmp_path / "h.json", per_task={"t1": 0.5, "t2": 0.5})
    with pytest.raises(ValueError, match="not deterministic"):
        paired_by_task(base, hand)


def test_refuses_a_run_without_per_task_scores(tmp_path):
    base = _write(tmp_path / "b.json", per_task={})
    hand = _write(tmp_path / "h.json", per_task={"t1": 0.5})
    with pytest.raises(ValueError, match="no per-task scores"):
        paired_by_task(base, hand)


def test_refuses_a_multi_seed_file(tmp_path):
    p = tmp_path / "m.json"
    arm = {"seed_heldout_by_task": {"t1": 0.5}, "opt_heldout_by_task": {"t1": 0.5}}
    p.write_text(json.dumps({"meta": {}, "runs": [{"A": arm}, {"A": arm}]}), encoding="utf-8")
    other = _write(tmp_path / "o.json", per_task={"t1": 0.5})
    with pytest.raises(ValueError, match="single-seed"):
        paired_by_task(str(p), other)


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15] * 4
    m, lo, hi = bootstrap_ci(diffs, b=2000, seed=0)
    again = bootstrap_ci(diffs, b=2000, seed=0)
    assert (m, lo, hi) == again  # fixed seed => reproducible to the digit
    assert lo <= m <= hi
    assert bootstrap_ci(diffs, b=2000, seed=1) != (m, lo, hi)  # and the seed matters


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError, match="no differences"):
        bootstrap_ci([])


def test_verdict_moves_when_a_large_effect_is_consistent():
    # 20 tasks all improving by 0.25: significant and far outside the SESOI.
    label, s = verdict([0.25] * 20)
    assert label == "MOVES"
    assert s["significant"] and not s["equivalent"]


def test_verdict_calls_a_consistent_but_tiny_effect_trivial():
    # Every task improves, so the sign test rejects, but the effect is below the SESOI.
    label, s = verdict([0.01] * 20)
    assert "TRIVIALLY" in label
    assert s["significant"] and s["equivalent"]


def test_verdict_reports_no_effect_only_when_the_interval_is_tight():
    label, s = verdict([0.0] * 8 + [0.01, -0.01] * 6)
    assert label.startswith("NO EFFECT")
    assert not s["significant"] and s["equivalent"]


def test_verdict_reports_inconclusive_when_underpowered():
    # Large scatter, no consistent sign: must NOT be reported as "no effect".
    label, s = verdict([0.5, -0.5, 0.4, -0.45, 0.5, -0.4])
    assert label.startswith("INCONCLUSIVE")
    assert not s["significant"] and not s["equivalent"]


def test_a_negative_effect_also_counts_as_moving():
    # C1's decision rule is two-sided: if the gene's prose can move held-out DOWN, the gene
    # still matters and an optimizer still has a gradient.
    label, s = verdict([-0.25] * 20)
    assert label == "MOVES"
    assert s["mean"] < 0


def test_sesoi_is_anchored_below_the_studys_own_measured_levers():
    # +0.063 (actionability) and +0.072 (tool-fed critique) are the effects the paper calls
    # real; the SESOI must sit below both or it would discard them as trivial.
    assert SESOI < 0.063
