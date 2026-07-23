"""Pins the paper's central mechanistic claim to the frozen cassettes.

The loop-level GEPA null is explained by the intrinsic critique being a
near-random failure detector; a non-oracle measurement tool fixes it. These
tests replay the committed cassettes (offline) and assert exactly that
qualitative contrast, so the claim can't silently rot.
"""

from __future__ import annotations

import pathlib
import sys

from whystuck.eval.text_stats import text_stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_critique_detector import collect_events, detector_stats  # noqa: E402

_C = "tests/fixtures/cassettes"


def test_intrinsic_critique_is_a_near_random_detector():
    ev = collect_events(f"{_C}/study-multihard.json", [0, 1, 2, 3, 4], loop_kw={})
    s = detector_stats(ev)
    assert s["n"] == 50  # 5 seeds x 10 held-out
    # near-random: it emits PASS on failing drafts more often than not
    assert s["precision"] < 0.5
    assert s["f1"] < 0.3


def test_measurement_tool_makes_the_critique_reliable():
    ev = collect_events(f"{_C}/study-feedback.json", [0, 1, 2], loop_kw={"stats_fn": text_stats})
    s = detector_stats(ev)
    assert s["n"] == 30
    # the tool-fed critique is a good detector, far better than intrinsic
    assert s["precision"] >= 0.9
    assert s["f1"] > 0.5


def test_tool_fed_f1_beats_intrinsic_f1():
    intrinsic = detector_stats(
        collect_events(f"{_C}/study-multihard.json", [0, 1, 2, 3, 4], loop_kw={})
    )
    tool_fed = detector_stats(
        collect_events(f"{_C}/study-feedback.json", [0, 1, 2], loop_kw={"stats_fn": text_stats})
    )
    assert tool_fed["f1"] > intrinsic["f1"] + 0.3
