"""Pins the evolved critique's PASS rate, and lets the split-size sweep be scored the same way.

The paper's load-bearing pre-registered prediction for the split sweep is about a PASS rate:
that the evolved critique keeps refusing to accept as the training split grows. That is a
property of a released gene, checkable offline for $0 -- but only if the scorer can be pointed
at a run other than the one it was written for, which is what these tests fix in place.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_evolved_gene import Condition, score_run  # noqa: E402

_C = "tests/fixtures/cassettes"


def test_the_evolved_critique_never_accepts_on_the_four_arm_rerun():
    """The mechanism behind arm D's loss at p0 = 0.067, and the paper's headline for it."""
    rows = score_run("docs/validation-study-c2b.json", f"{_C}/study-c2b.json")
    evolved = [r for r in rows if r.kind == "evolved"]
    assert len(evolved) == 3  # one per seed
    assert all(r.pass_rate == 0.0 for r in evolved)
    assert sum(r.stats["tp"] for r in evolved) == 0


def test_the_seed_rubric_does_accept_so_zero_is_a_finding_not_a_default():
    """If the seed rubric also never emitted PASS, a PASS rate of 0 would say nothing."""
    rows = score_run("docs/validation-study-c2b.json", f"{_C}/study-c2b.json")
    seed = [r for r in rows if r.kind == "seed"]
    assert all(r.pass_rate > 0 for r in seed)
    assert sum(r.stats["tp"] for r in seed) == 1


def test_score_run_rejects_a_results_cassette_mismatch():
    """A silently-empty replay would report 'no events' as 'no PASSes', the same number the
    finding predicts, so a mismatch has to raise.

    It raises `CassetteMiss` rather than the scorer's own guard, which is the Evidence-7 fix
    doing its job one layer down: `collect_events` skips an un-recorded task (so a superset of
    seeds is safe to pass), but the cassette refuses to exit cleanly once a miss has been
    swallowed. Asserting the stronger of the two is the point -- the scorer's guard is the
    backstop for an empty-but-hit replay, not the primary defence.
    """
    from whystuck.eval.cassette import CassetteMiss

    with pytest.raises(CassetteMiss, match="had no recorded output"):
        score_run("docs/validation-study-c2b.json", f"{_C}/study-p0-L2.json")


def test_condition_is_addressable_per_seed():
    """The sweep needs per-gene PASS rates, not a pooled one: its prediction is stated as
    'at least five of six genes', so the unit has to survive to the caller."""
    rows = score_run("docs/validation-study-c2b.json", f"{_C}/study-c2b.json")
    assert {r.seed for r in rows} == {0, 1, 2}
    assert all(isinstance(r, Condition) for r in rows)
    assert all(r.n == 10 for r in rows)
