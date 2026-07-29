"""The study's missing variance component, recovered from runs it already bought.

Every interval in this paper describes task-draw and search-path variation and explicitly
*not* measurement noise: the worker runs at the provider's default temperature, one sample per
task, so nothing was ever measured twice. That was stated as a scope note. It is also a gap,
because several released runs score the **same seed genome** on the **same held-out tasks** --
the four-arm re-run and each row of the split-size sweep all evaluate the unoptimized genome
before optimizing anything. Those are independent draws of one fixed quantity, so their spread
is exactly the missing component, and it costs nothing to extract.

The tests below pin two things: the estimate, and the guard that makes it trustworthy. The
guard matters more than the number. Task ids are generated positionally, so two runs with
*different constraint pools* carry the *same* ids -- `c2b` and the p0 levels have byte-identical
held-out id sets while scoring genuinely different tasks. Pooling those would book a difficulty
difference as noise and inflate the estimate. An id check cannot see it; the payloads and the
recorded exclusions can.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_measurement_noise import (  # noqa: E402
    IncompatibleReplicates,
    eligible_replicates,
    noise_estimate,
    seed_vectors,
)


def test_eligible_replicates_are_the_no_lever_no_exclusion_multihard_runs():
    names = {r.name for r in eligible_replicates()}
    # The four-arm re-run plus whichever split-sweep rows have been bought.
    assert "c2b" in names
    assert names & {"split4", "split16", "split24"}
    # None of these may carry a lever or an exclusion: both change what the seed loop scores.
    for r in eligible_replicates():
        assert r.exclusions == frozenset()
        assert r.levers == ()


def test_the_p0_levels_are_excluded_despite_identical_task_ids():
    """The trap this module exists to avoid, asserted rather than trusted."""
    names = {r.name for r in eligible_replicates()}
    assert "p0-L2" not in names
    assert "p0-L3" not in names


def test_pooling_runs_with_different_exclusions_raises():
    with pytest.raises(IncompatibleReplicates, match="exclusion"):
        seed_vectors(["c2b", "p0-L2"])


def test_identical_task_ids_are_not_treated_as_identical_tasks():
    """If this ever passes by id alone, the guard has stopped working."""
    from analyze_measurement_noise import heldout_ids

    assert heldout_ids("c2b") == heldout_ids("p0-L2"), (
        "ids are expected to collide -- that is the whole hazard; if they stop colliding "
        "this test is asleep and the payload check is no longer load-bearing"
    )


def test_noise_estimate_is_positive_and_smaller_than_the_effects_it_qualifies():
    est = noise_estimate()
    assert est.n_replicates >= 2
    assert est.n_tasks == 30  # 3 seeds x 10 held-out
    # A fixed genome re-scored must move, or the metric would be deterministic and the
    # study's "no sampling variance" note would be a statement about nature, not scope.
    assert est.sigma_single > 0
    # And it must be small enough that the paper's effects are not pure noise, while being
    # large enough to qualify them. Both directions are the finding.
    assert 0.01 < est.sigma_delta_mean < 0.06


def test_a_signed_mean_difference_is_reported_not_hidden():
    """Re-measuring one genome should average to ~0. A drift term would show up here, and
    it is confounded with sampling noise across runs recorded on different days."""
    est = noise_estimate()
    assert abs(est.mean_signed) < 3 * est.sigma_delta_mean


def test_effects_are_reported_in_units_of_the_noise():
    est = noise_estimate()
    # The paper's own headline numbers, in sigma. None of them should silently exceed 3.
    for delta in (-0.0417, 0.0600, 0.0617):
        assert est.in_sigma(delta) < 3.0
