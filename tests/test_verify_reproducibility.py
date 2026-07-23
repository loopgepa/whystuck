"""Guards for the reproducibility harness itself.

A reproducibility checker that can pass vacuously is worse than none. These pin
the two properties that keep it honest: it declares success only after verifying
every registered run, and its score comparison actually discriminates. (Full
replay of the real cassettes needs gepa+litellm and lives in the eval venv, run
via `scripts/verify_reproducibility.py`; these are the gepa-free logic guards.)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import verify_reproducibility as vr  # noqa: E402


def test_runs_list_registers_all_expected_runs():
    assert len(vr.RUNS) == vr.EXPECTED_RUNS == 10


def test_main_does_not_vacuously_succeed_on_empty_runs(monkeypatch, capsys):
    # An empty/truncated RUNS list must NOT print the success banner or exit 0 —
    # `all_ok` seeds True, so the loop-only AND would otherwise claim reproduction.
    monkeypatch.setattr(vr, "RUNS", [])
    code = vr.main()
    out = capsys.readouterr().out
    assert code != 0
    assert "REPRODUCE FROM CASSETTE" not in out
    assert "INCOMPLETE" in out


def _mk(**arm_vals):
    # a runs entry: every arm A-D with (seed, opt) held-out scores
    return {a: {"seed_heldout": v, "opt_heldout": v} for a, v in arm_vals.items()}


def test_scores_comparison_discriminates_a_perturbed_result():
    base = [_mk(A=0.5, B=0.5, C=0.5, D=0.5)]
    same = [_mk(A=0.5, B=0.5, C=0.5, D=0.5)]
    perturbed = [_mk(A=0.5, B=0.5, C=0.5, D=0.6)]  # one arm differs
    assert vr._scores(base) == vr._scores(same)
    assert vr._scores(base) != vr._scores(perturbed)
