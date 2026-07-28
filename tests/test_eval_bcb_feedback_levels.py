"""Tests for the code-domain feedback-actionability lever (bcb ``_shape_feedback``).

The IFBench study found (runs 9--10) that measurement-shaped, concrete repair feedback
lifts the self-refine baseline while a terse verdict does not. This is the cross-domain
analogue for code: the in-loop signal from the VISIBLE unittest methods is shaped at three
levels of actionability -- ``lo`` (verdict only), ``mid`` (the raw failure detail, the
current default), and ``hi`` (the failure detail PLUS the source of the visible tests the
solution must satisfy). Only visible-method source is ever shown; the hidden scorer is not.

Pure formatting (no Docker), so these run everywhere.
"""

from __future__ import annotations

from whystuck.eval.bcb_bench import _method_sources, _shape_feedback

_TEST_SRC = (
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_sum(self):\n        self.assertEqual(task_func([1, 2]), 3)\n"
    "    def test_empty(self):\n        self.assertEqual(task_func([]), 0)\n"
    "    def test_hidden_only(self):\n        self.assertEqual(task_func([9]), 9)\n"
)
_VISIBLE = ["test_sum", "test_empty"]


def test_pass_message_is_identical_across_levels():
    # A pass says the same thing regardless of level; the lever only shapes FAILURES.
    kw = dict(passed=True, n=2, detail="", test_src=_TEST_SRC, visible_methods=_VISIBLE)
    msgs = {lvl: _shape_feedback(lvl, **kw) for lvl in ("lo", "mid", "hi")}
    assert len(set(msgs.values())) == 1
    assert "pass" in msgs["lo"]


def test_lo_is_verdict_only_no_detail():
    out = _shape_feedback(
        "lo",
        passed=False,
        n=2,
        detail="AssertionError: 1 != 3",
        test_src=_TEST_SRC,
        visible_methods=_VISIBLE,
    )
    assert "AssertionError" not in out  # the concrete detail is withheld at lo
    assert "revise" in out.lower()


def test_mid_is_the_raw_detail_and_matches_legacy_string():
    # mid must be byte-identical to the pre-lever feedback so the recorded cassette/tests hold.
    out = _shape_feedback(
        "mid",
        passed=False,
        n=2,
        detail="AssertionError: 1 != 3",
        test_src=_TEST_SRC,
        visible_methods=_VISIBLE,
    )
    assert out == "A visible test failed:\nAssertionError: 1 != 3"


def test_hi_adds_visible_test_source_but_not_hidden():
    out = _shape_feedback(
        "hi",
        passed=False,
        n=2,
        detail="AssertionError: 1 != 3",
        test_src=_TEST_SRC,
        visible_methods=_VISIBLE,
    )
    assert "AssertionError: 1 != 3" in out  # keeps the mid detail
    assert "def test_sum" in out and "def test_empty" in out  # adds visible source
    assert "test_hidden_only" not in out  # never leaks the non-visible method
    assert len(out) > len(
        _shape_feedback(
            "mid",
            passed=False,
            n=2,
            detail="AssertionError: 1 != 3",
            test_src=_TEST_SRC,
            visible_methods=_VISIBLE,
        )
    )


def test_method_sources_extracts_only_named_methods():
    src = _method_sources(_TEST_SRC, ["test_sum"])
    assert "def test_sum" in src
    assert "def test_empty" not in src and "def test_hidden_only" not in src


def test_method_sources_survives_unparseable_test():
    assert _method_sources("def broken(: pass", ["test_sum"]) == ""
