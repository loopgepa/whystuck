"""Pins the fix-step residual decomposition to the frozen cassettes.

The paper's closing mechanistic claim is that the generator-limited residual (the
gap the loop can't close even with a perfect detector and maximally-actionable
feedback) is almost entirely ONE constraint — exact_word_count — while the
"choice" constraints are solved once feedback is actionable. These tests replay
the committed cassettes offline and assert exactly that, so the claim can't rot.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_residual_by_constraint import constraint_pass_counts, pass_rate  # noqa: E402

_C = "tests/fixtures/cassettes"


def test_exact_word_count_is_the_binding_residual_under_perfect_actionable_feedback():
    # run 10: oracle detector (perfect) + concrete measured edits (hi), R=3.
    c = constraint_pass_counts(
        f"{_C}/study-fbacthi.json",
        [0, 1, 2],
        oracle_critique=True,
        feedback_level="hi",
        max_retries=3,
    )
    # exact_word_count stays the hard one even when told exactly how many words to add.
    assert pass_rate(c, "exact_word_count") < 0.7
    # every "choice" constraint is solved once feedback is actionable.
    for iid in ("no_comma", "all_lowercase", "num_bullets", "include_keyword"):
        assert pass_rate(c, iid) == 1.0, iid


def test_actionable_feedback_lifts_exact_word_count_over_intrinsic():
    intrinsic = constraint_pass_counts(f"{_C}/study-multihard.json", [0, 1, 2, 3, 4], max_retries=1)
    hi = constraint_pass_counts(
        f"{_C}/study-fbacthi.json",
        [0, 1, 2],
        oracle_critique=True,
        feedback_level="hi",
        max_retries=3,
    )
    # the binding constraint improves with actionable feedback, but never gets solved.
    assert pass_rate(hi, "exact_word_count") > pass_rate(intrinsic, "exact_word_count")
    assert pass_rate(hi, "exact_word_count") < 1.0


def test_pass_rate_arithmetic_on_known_counts():
    counts = {"a": [3, 4], "b": [0, 0]}
    assert pass_rate(counts, "a") == 0.75
    assert pass_rate(counts, "missing") != pass_rate(counts, "missing")  # nan (total 0)
    assert pass_rate(counts, "b") != pass_rate(counts, "b")  # nan, not a ZeroDivisionError
