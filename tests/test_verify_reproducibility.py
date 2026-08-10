"""Guards for the reproducibility harness itself.

A reproducibility checker that can pass vacuously is worse than none. These pin
the two properties that keep it honest: it declares success only after verifying
every registered run, and its score comparison actually discriminates. (Full
replay of the real cassettes needs gepa+litellm and lives in the eval venv, run
via `scripts/verify_reproducibility.py`; these are the gepa-free logic guards.)

A third property arrived later and is the one this file exists for now: SCOPE. Every
released run backs at least one published number -- instrumenting the claim checker shows it
reads all 43 -- but for a long while only 26 were registered here, so 17 runs supplied paper
values while holding no replay status at all. Nothing was wrong with them; eleven of them
replay bit-for-bit, which is precisely the point. Their status was unstated, and an unstated
status reads to a reader as the guarantee covering everything it does not disclaim. That is
this study's own subject applied to its release, so it gets a machine check rather than
diligence: the partition below must stay total.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import verify_reproducibility as vr  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _released() -> set[str]:
    return {f"docs/{p.name}" for p in (ROOT / "docs").glob("validation-study*.json")}


def _registered() -> set[str]:
    return {committed for _label, committed, *_ in vr.RUNS}


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


def test_every_released_run_has_a_recorded_replay_status():
    """No released run may hold no status. This is the scope guard.

    Three states, and the union must be everything in `docs/`: registered in `RUNS` (it
    replays, or it is `SUPERSEDED` with a stated reason), or listed in `UNREPLAYABLE` with a
    stated reason. A run that is in neither is not "known to be fine" -- it is a run whose
    reproducibility nobody has claimed either way, while its numbers sit in a paper.
    """
    unaccounted = sorted(_released() - _registered() - set(vr.UNREPLAYABLE))
    assert not unaccounted, (
        f"these released runs back paper numbers but hold no replay status: {unaccounted}. "
        "Register them in verify_reproducibility.RUNS (with frozen CLI args), or add them to "
        "UNREPLAYABLE with a reason no replay can be produced. Do not leave them silent."
    )


def test_no_run_is_claimed_both_replayable_and_unreplayable():
    """The two lists must be disjoint, or a run's status depends on which one you read."""
    both = sorted(_registered() & set(vr.UNREPLAYABLE))
    assert not both, f"registered as replayable AND listed unreplayable: {both}"


def test_no_unreplayable_entry_names_a_run_that_does_not_exist():
    """A dead exclusion stops describing the release and starts excusing whatever collides
    with its path next -- the same rot `test_paper_table_coverage._EXEMPT` is guarded against.

    `RUNS` is deliberately NOT held to this: a row may be registered before it is bought, which
    is how the exact args get committed ahead of the numbers. An exclusion has no such excuse.
    """
    dead = sorted(p for p in vr.UNREPLAYABLE if not (ROOT / p).exists())
    assert not dead, f"UNREPLAYABLE excuses runs that are not in the release: {dead}"


def test_every_stated_reason_is_an_actual_reason():
    """Guard against the escape hatch: satisfying the partition with an empty string.

    A one-word reason passes the letter of the scope guard and defeats its purpose, so the
    floor is a sentence's worth of text. This is a crude proxy and is meant to be -- it cannot
    tell a good reason from a bad one, only a stated one from a placeholder.
    """
    thin = sorted(
        k for d in (vr.SUPERSEDED, vr.UNREPLAYABLE) for k, v in d.items() if len(v.strip()) < 40
    )
    assert not thin, f"these entries state a placeholder rather than a reason: {thin}"


def test_the_scope_guard_is_not_vacuous():
    """If `_released()` ever returned nothing, the partition test above passes forever."""
    released = _released()
    assert len(released) >= 40, f"expected the full release, found {len(released)} run JSONs"
    assert vr.UNREPLAYABLE, "no exclusions listed; delete the guard or restore the entries"
    assert _registered() & released, "no registered run names a released artifact"


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
    monkeypatch.setattr(vr, "SUPERSEDED", {})
    rc = vr.main()
    out = capsys.readouterr().out
    assert "PENDING" in out, out
    assert "FAIL" not in out, "an absent artifact is not a reproduction failure"
    assert rc != 0, "nothing was verified, so the script must not report success"
    assert "0 replayable run(s) checked" in out or "checked" in out
