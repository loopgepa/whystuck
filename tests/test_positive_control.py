"""The positive control must keep passing, or the paper's nulls stop meaning anything.

Every headline result in the study is a near-zero effect. A near-zero effect is only
informative if the instrument can register a non-zero one, and this project has now found
five separate defects that produced confident zeros from experiments that did not run. So
"the harness detects a loop-level advantage when one exists" is pinned as a test rather than
asserted in prose.

Skipped in the main venv, which deliberately has no gepa (a separate test asserts that the
absence produces an actionable error), and the eval venv that does have gepa has no pytest.
So in practice the control is executed as a script:

    scratchpad/evalvenv/bin/python scripts/positive_control.py

These cases are the guard for when the two environments meet — and the first of them checks
the landscape itself, which needs no optimizer and therefore runs wherever gepa is present.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("gepa", reason="the positive control drives real GEPA; eval venv only")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from positive_control import MIN_GAP, SENTINEL, _reflection, _worker  # noqa: E402


def test_landscape_makes_the_verify_gene_the_only_lever():
    # Guard the control itself: if the landscape stopped being rigged, the control would
    # pass or fail for reasons unrelated to the harness.
    assert _worker("write something") == "x x", "the draft must fail the constraint"
    assert _worker("# Draft answer\nx x") == "PASS", "a seed critique must miss the flaw"
    caught = _worker(f"# Draft answer\nx x\n{SENTINEL}")
    assert caught.startswith("FAIL"), "an optimized critique must catch the flaw"
    assert _worker("# Problems to fix\nwhitespace") == "solid", "a refine must reach a pass"
    assert SENTINEL in _reflection("anything"), "the proposal must carry the sentinel"


def test_the_harness_detects_an_advantage_that_exists():
    from positive_control import main

    assert main() == 0, (
        "the four-arm study did not register a loop-level advantage that is true by "
        f"construction (needs Δ(D−B) ≥ {MIN_GAP}); until it does, no null in the paper "
        "can be read as evidence of absence"
    )
