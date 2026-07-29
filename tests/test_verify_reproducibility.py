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


def test_every_registered_run_is_marked_superseded():
    # Eleven recorded runs predate the harness correction (component-agnostic reflective
    # dataset + one-component-per-proposal selector + saturated arm C). The corrected
    # wiring changes every reflection prompt, hence every cassette key, so none of them
    # can replay under the current code. EXPECTED_RUNS counts runs recorded under the
    # CORRECTED harness: the two C1 conditions and the C2 four-arm re-run.
    # Pin the INVARIANT, not the snapshot: every run recorded under the pre-correction
    # harness stays superseded, and EXPECTED_RUNS must equal however many runs are not.
    # The previous version asserted "11 runs, all superseded, EXPECTED_RUNS == 0", which
    # was a true fact about a moment rather than a rule, and broke the moment the first
    # corrected-harness run was legitimately registered.
    labels = {label for label, *_ in vr.RUNS}
    assert set(vr.SUPERSEDED) <= labels, "a superseded label names no registered run"
    live = labels - set(vr.SUPERSEDED)
    assert len(live) == vr.EXPECTED_RUNS, (
        f"EXPECTED_RUNS={vr.EXPECTED_RUNS} but {len(live)} run(s) are not superseded: "
        f"{sorted(live)}; a stale count lets a lost run pass unnoticed"
    )


def test_all_superseded_does_not_claim_reproduction(monkeypatch, capsys):
    # The dangerous case: every run skipped, no failures recorded, so a naive
    # `all_ok and verified >= EXPECTED_RUNS` would exit 0 and read as "reproduction
    # proven" while nothing at all was replayed. Constructed by monkeypatch rather than
    # read off the real registry, so it keeps testing that guarantee after a
    # corrected-harness run is registered.
    monkeypatch.setattr(vr, "RUNS", [(label, "x.json", "x.json", []) for label in vr.SUPERSEDED])
    monkeypatch.setattr(vr, "EXPECTED_RUNS", 0)
    code = vr.main()
    out = capsys.readouterr().out
    assert code != 0
    assert "REPRODUCE FROM CASSETTE" not in out
    assert "NOTHING PROVEN" in out


def test_main_does_not_vacuously_succeed_on_empty_runs(monkeypatch, capsys):
    # An empty/truncated RUNS list must NOT print the success banner or exit 0 —
    # `all_ok` seeds True, so the loop-only AND would otherwise claim reproduction.
    monkeypatch.setattr(vr, "RUNS", [])
    code = vr.main()
    out = capsys.readouterr().out
    assert code != 0
    assert "REPRODUCE FROM CASSETTE" not in out
    assert "NOTHING PROVEN" in out


def _mk(**arm_vals):
    # a runs entry: every arm A-D with (seed, opt) held-out scores
    return {a: {"seed_heldout": v, "opt_heldout": v} for a, v in arm_vals.items()}


def test_scores_comparison_discriminates_a_perturbed_result():
    base = [_mk(A=0.5, B=0.5, C=0.5, D=0.5)]
    same = [_mk(A=0.5, B=0.5, C=0.5, D=0.5)]
    perturbed = [_mk(A=0.5, B=0.5, C=0.5, D=0.6)]  # one arm differs
    assert vr._scores(base) == vr._scores(same)
    assert vr._scores(base) != vr._scores(perturbed)


def test_a_registered_but_unbought_run_is_pending_not_a_failure(monkeypatch, capsys):
    """A registry row whose results artifact does not exist yet must read as PENDING.

    The registry deliberately carries rows for pre-registered runs before they are bought, so
    the exact CLI args are committed ahead of the numbers. Reporting an absent artifact as FAIL
    conflates "not bought yet" with "does not reproduce" -- the same infra-versus-data
    conflation this study spent an appendix on. Pending rows must also NOT count toward
    `verified`, so the script still refuses to claim completeness.
    """
    monkeypatch.setattr(
        vr,
        "RUNS",
        [("future run", "docs/does-not-exist-yet.json", "tests/fixtures/nope.json", [])],
    )
    monkeypatch.setattr(vr, "SUPERSEDED", frozenset())
    rc = vr.main()
    out = capsys.readouterr().out
    assert "PENDING" in out, out
    assert "FAIL" not in out, "an absent artifact is not a reproduction failure"
    assert rc != 0, "nothing was verified, so the script must not report success"
    assert "0 replayable run(s) checked" in out or "checked" in out
