"""A Δ(D−B) is only a D-vs-B contrast if BOTH arms received their treatment.

This paper retracted eleven results because arm D's verify gene never changed, and built a
budget guard so a starved arm is refused before it bills. The guard watches arm D. It does not
watch arm B — and the split-size sweep's smallest row shows why that matters: at
`n_train = 4`, arm B accepted nothing on three of three seeds, so its Δ compares loop-level
optimization against an *unoptimized seed genome* rather than against prompt-only
optimization. That is the same defect as the retraction, entering from the unwatched side.

So the check is symmetric here: for every released contrast, count the seeds on which each
optimizing arm actually changed a gene, and refuse to let a row be read as D-vs-B unless both
did. The paper's own reporting minimum — per component, whether the optimizer's output differs
from the seed — applied to both arms instead of one.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_deltas import arms_optimized, is_valid_contrast  # noqa: E402

_DOCS = "docs"


def test_the_four_arm_rerun_is_a_valid_contrast():
    """C2b is the run the paper's headline rests on, so both arms must have optimized."""
    counts = arms_optimized(f"{_DOCS}/validation-study-c2b.json")
    assert counts["D"] == 3
    assert counts["B"] == 2  # generate accepted on two of three seeds
    assert is_valid_contrast(f"{_DOCS}/validation-study-c2b.json")


@pytest.mark.parametrize("level", ["p0-L2", "p0-L3"])
def test_the_p0_levels_are_valid_contrasts(level):
    """The sign flip is the paper's central claim; it cannot rest on a one-sided contrast."""
    counts = arms_optimized(f"{_DOCS}/validation-study-{level}.json")
    assert counts["B"] > 0, "prompt-only accepted nothing: this is not a D-vs-B contrast"
    assert counts["D"] > 0
    assert is_valid_contrast(f"{_DOCS}/validation-study-{level}.json")


def test_a_one_sided_contrast_is_reported_as_invalid():
    """The smallest split-size row, which is the reason this check exists.

    Skipped rather than asserted-away when the run is not present: the row is bought
    separately, and a check that silently passes on a missing file is worse than no check.
    """
    path = f"{_DOCS}/validation-study-split4.json"
    if not pathlib.Path(path).exists():
        pytest.skip(f"{path} not present — the n_train=4 sweep row has not been bought")
    counts = arms_optimized(path)
    assert counts["D"] == 3
    assert counts["B"] == 0
    assert not is_valid_contrast(path), (
        "arm B accepted nothing on any seed, so this row's Δ compares loop-level "
        "optimization against the seed genome, not against prompt-only optimization"
    )


def test_arms_optimized_only_counts_arms_that_may_evolve():
    """Arms A and C are not asked to change anything, so they must not appear as failures."""
    counts = arms_optimized(f"{_DOCS}/validation-study-c2b.json")
    assert set(counts) == {"B", "D"}


def test_a_run_with_no_acceptance_record_is_not_reported_as_accepting_nothing():
    """`arms_optimized` returns zeros for a run recorded before `genes_changed` existed.

    Fifteen released runs are in that position, and this study spent months reading their zeros as
    measurements. The counting behaviour stays, because callers depend on it; what changes is that
    `acceptance_recorded` lets a caller tell the two apart, and `verify_paper_claims` now counts
    them separately so Paper III's "by an absent field rather than a measured one" is derived.
    """
    from analyze_deltas import acceptance_recorded

    pre = f"{_DOCS}/validation-study-multihard.json"
    post = f"{_DOCS}/validation-study-c2b.json"
    assert not acceptance_recorded(pre), (
        "multihard predates genes_changed; if it now records acceptance, the run set changed and "
        "the fifteen-run count in Paper III needs re-deriving"
    )
    assert acceptance_recorded(post), "the four-arm re-run records acceptance"
    # The trap itself: zeros from a run that recorded nothing.
    counts = arms_optimized(pre)
    assert counts and all(v == 0 for v in counts.values())
    assert not is_valid_contrast(pre), "a run with no acceptance record is not a valid contrast"
