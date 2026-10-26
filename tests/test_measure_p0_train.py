"""p0 on the split the optimizer trains on, beside the held-out p0 the papers tabulate.

Paper I's mechanism is a training-signal argument: an always-refine critique is nearly free on a
training signal where almost nothing is already correct. The p0 column it prints is measured on
HELD-OUT first drafts. The two differ (at L2 the train split's base rate is 1.7x the held-out
one), so the train-side value is derived here rather than assumed equal.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from measure_p0 import TRAIN_LEVELS, train_p0  # noqa: E402


def test_train_split_p0_per_level():
    got = {label: train_p0(cassette, exclude) for label, cassette, exclude in TRAIN_LEVELS}
    assert got == {"L1": (3, 24), "L2": (11, 24), "L3": (17, 24)}


def test_train_p0_raises_on_a_cassette_miss(tmp_path):
    """An absent recording must not read as a smaller sample."""
    import json

    import pytest

    from whystuck.eval.cassette import CassetteMiss

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"entries": {}}))
    with pytest.raises(CassetteMiss):
        train_p0(str(empty), frozenset())
