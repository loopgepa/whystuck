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

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_measurement_noise import (  # noqa: E402
    WORKER_MODEL,
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


def test_a_run_on_a_different_worker_is_not_a_replicate():
    """The regime includes the WORKER, and eligibility did not check it.

    Noise here is a property of a genome executed by a particular model: sigma is measured by
    re-scoring one fixed seed genome, so a run whose worker is a different model is measuring a
    different quantity entirely. Eligibility screened task set, seed count, exclusions, levers
    and gene overrides -- and not the worker -- so a three-seed `llama-3.1-8b` study dropped into
    `docs/` silently enrolled as an eleventh replicate and moved every published sigma. The
    claim checker caught it, 29 checks at once, which is the only reason it is a test and not a
    correction.
    """
    for r in eligible_replicates():
        meta = json.loads(pathlib.Path(r.path).read_text())["meta"]
        assert meta.get("worker_model") == WORKER_MODEL, (
            f"{r.name} runs on {meta.get('worker_model')!r}, not the regime's {WORKER_MODEL!r}; "
            "its noise is not this genome's noise"
        )


def test_a_spliced_run_is_not_a_replicate():
    """A verdict/prose splice issues two critique calls, so its loop is not the seed loop."""
    for r in eligible_replicates():
        meta = json.loads(pathlib.Path(r.path).read_text())["meta"]
        assert not meta.get("verdict_model"), f"{r.name} is spliced; its loop differs"
        assert not meta.get("critique_model"), f"{r.name} routes the critique elsewhere"


def test_the_contamination_moved_exactly_the_number_of_values_the_paper_claims():
    """Paper III's defect 11 states that the un-screened estimator moved 29 published values.

    That figure is in a caption and in the body, so it is a claim like any other and it rots like
    any other. It is derived here by re-running the claim checker with the screens removed and
    counting which checks change, rather than by remembering what the failure output said.
    """
    import functools
    import re

    import analyze_measurement_noise as amn
    import verify_paper_claims as vpc

    def clear_caches():
        for mod in (vpc, amn):
            for obj in vars(mod).values():
                if callable(obj) and hasattr(obj, "cache_clear"):
                    obj.cache_clear()

    def actuals():
        clear_caches()
        return {c[0]: (c[1], c[2], c[3]) for c in vpc._checks()}

    before = actuals()

    original = amn.eligible_replicates

    @functools.wraps(original)
    def unscreened():
        out = []
        for path in sorted(amn.glob.glob(f"{amn.DOCS}/validation-study*.json")):
            doc = amn.json.loads(amn.Path(path).read_text())
            meta, runs = doc["meta"], doc["runs"]
            if meta.get("task_set") != amn._TASK_SET or meta.get("n_heldout") != amn._N_HELDOUT:
                continue
            if len(runs) != amn._N_SEEDS:
                continue
            arm = next((a for a in "ABCD" if isinstance(runs[0].get(a), dict)), None)
            if arm is None or runs[0][arm].get("seed_heldout_by_task") is None:
                continue
            if frozenset(meta.get("exclude_constraint") or ()):
                continue
            if tuple(k for k in amn._LEVERS if meta.get(k)):
                continue
            if tuple(sorted(meta.get("gene_overrides") or ())):
                continue
            out.append(amn.Replicate(amn._name_of(path), path, frozenset(), (), ()))
        return out

    amn.eligible_replicates = unscreened
    try:
        contaminated = unscreened()
        assert any("splice" in r.name for r in contaminated), (
            "the contaminating run is no longer in docs/, so this test measures nothing; "
            "it lived at docs/validation-study-splice-G1.json"
        )
        after = actuals()
    finally:
        amn.eligible_replicates = original
        clear_caches()

    moved = [k for k in before if k in after and before[k][2] != after[k][2]]
    # Of those, the ones that moved FAR ENOUGH for their own tolerance to notice. The gap between
    # the two counts is the point: the rest shifted silently inside tolerance.
    caught = [k for k in moved if abs(after[k][2] - after[k][0]) > after[k][1]]

    tex = pathlib.Path("docs/papers/p3-silent-no-op/main.tex").read_text(encoding="utf-8")
    m = re.search(
        r"It moves \$(\d+)\$ of this paper's published\s+values, and only \$(\d+)\$ of the",
        tex,
    )
    assert m, "Paper III no longer states the two contamination counts in the form this guard reads"
    claimed_moved, claimed_caught = int(m.group(1)), int(m.group(2))
    assert len(moved) == claimed_moved, (
        f"the paper says {claimed_moved} values moved; re-deriving gives {len(moved)}"
    )
    assert len(caught) == claimed_caught, (
        f"the paper says {claimed_caught} of them breached tolerance; re-deriving gives "
        f"{len(caught)}"
    )
    assert len(caught) < len(moved), (
        "if every moved value breached tolerance the sentence loses its point, which is that "
        "some shifted without any guard noticing"
    )
