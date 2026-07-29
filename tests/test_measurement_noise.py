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


def test_uniform_drift_cancels_but_task_specific_drift_does_not():
    """The property the paper's interpretation of sigma rests on.

    The estimator takes `pstdev` of the PER-TASK difference vector between two replicates, and
    pstdev subtracts the mean. So a provider shift that moves every task by the same amount --
    uniform drift -- cancels exactly, and sigma measures sampling noise cleaned of it. Only
    TASK-SPECIFIC drift inflates sigma. Getting this backwards would understate the estimate's
    trustworthiness in one direction and overstate it in the other, and an earlier draft of
    sec:threats said only "confounded with drift" without the distinction.

    The uniform component is not lost, it is reported separately as `mean_signed`.
    """
    import itertools
    from statistics import mean, pstdev

    names = [r.name for r in eligible_replicates()]
    if len(names) < 2:
        pytest.skip("need at least two eligible replicates")
    vectors = seed_vectors(names)

    def sigma_of(vecs: dict[str, list[float]]) -> float:
        sds = [
            pstdev([y - x for x, y in zip(vecs[a], vecs[b], strict=True)])
            for a, b in itertools.combinations(names, 2)
        ]
        return mean(sds) / 2**0.5

    base = sigma_of(vectors)

    uniform = {k: ([x + 0.1 for x in v] if k == names[0] else list(v)) for k, v in vectors.items()}
    assert sigma_of(uniform) == pytest.approx(base, abs=1e-12), (
        "a uniform shift must cancel: pstdev of the difference removes its mean"
    )

    # Task-specific drift: alternating tasks in opposite directions, so the MEAN shift is still
    # zero and only the per-task pattern changes. The amplitude is deliberately large (0.3): only
    # the pairs involving the perturbed replicate change, so a small amplitude is diluted by the
    # unperturbed pairs and would make this assert a threshold rather than the property.
    def alternate(v: list[float]) -> list[float]:
        return [x + (0.3 if i % 2 else -0.3) for i, x in enumerate(v)]

    specific = {k: (alternate(v) if k == names[0] else list(v)) for k, v in vectors.items()}
    assert sigma_of(specific) > base * 1.2, (
        "task-specific drift MUST inflate sigma, or the estimator would be blind to it"
    )

    # And the uniform component is still reported, just elsewhere.
    est = noise_estimate(names)
    assert abs(est.mean_signed) < 0.05, "a re-measurement's mean signed difference should be small"
