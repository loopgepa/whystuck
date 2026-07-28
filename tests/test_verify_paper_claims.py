"""The paper's numbers must keep matching the released artifacts.

A fact-check pass found two errors that had survived several readings (three runs
overrode the $5 default when four did; "registers eleven runs" after two more were added),
so agreement between the paper and its data is a test rather than a memory.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from verify_paper_claims import _checks  # noqa: E402


@pytest.mark.parametrize("label,claimed,tol,actual", _checks(), ids=lambda v: str(v)[:40])
def test_paper_claim_matches_the_artifact(label, claimed, tol, actual):
    assert abs(actual - claimed) <= tol, (
        f"{label}: the paper says {claimed} but the released artifacts give {actual}"
    )


def test_the_checklist_is_not_silently_empty():
    # A refactor that broke _checks() would otherwise make this file pass vacuously,
    # which is the same failure mode the paper is about.
    assert len(_checks()) >= 50
