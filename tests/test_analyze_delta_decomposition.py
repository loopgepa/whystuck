"""Pins the component-level decomposition of Δ(D−B) to the frozen cassettes.

Δ(D−B) is a *whole-system* contrast, but only `ensure_correct` is the treatment:
`generate` is evolvable in BOTH arms, so part of any gap is the shared gene rather than
the loop-level intervention. These tests pin the split — Δ = draft term + loop term — and
the pre-treatment stratification that shows which drafts the gap lives on.

They exist because the paper drew a mechanism from the aggregate that the decomposition
contradicts: the loss at L1 was attributed to a never-accepting critique damaging drafts
that needed no repair, and the already-correct stratum turns out to contribute the
opposite sign. A claim that specific has to be a test, not a reading.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_delta_decomposition import (  # noqa: E402
    LEVELS,
    SEED_ARM,
    SPLIT_LEVELS,
    answer_length,
    decompose,
    refine_productivity,
    released_delta,
    strata,
)

# Measured from the committed cassettes. Every figure the paper prints for this
# decomposition is pinned here, so a reworded sentence in main.tex cannot drift from the
# artifact without a test going red.
#
# tol is 5e-4: the paper reports these to three decimals.
_TOL = 5e-4

#: level -> (delta, draft_term, loop_term)
_EXPECTED = {
    "L1": (-0.041667, -0.025000, -0.016667),
    "L2": (+0.060000, +0.000000, +0.060000),
    "L3": (-0.001667, +0.005000, -0.006667),
}

#: level -> stratum -> (n, mean_B, mean_D). Four decimals, matching the paper's table:
#: three would put 0.8875 and 0.9775 exactly on a rounding boundary, and a claim that
#: only holds under a particular rounding rule is not a claim worth pinning.
_STRATA_TOL = 1e-4
_STRATA = {
    "L1": {True: (2, 0.7750, 0.8000), False: (28, 0.7893, 0.7429)},
    "L2": {True: (8, 0.8875, 1.0000), False: (22, 0.9295, 0.9705)},
    "L3": {True: (20, 0.9775, 0.9675), False: (10, 0.9600, 0.9750)},
}


@pytest.fixture(scope="module")
def decomposed() -> dict[str, object]:
    return {lvl.name: decompose(lvl) for lvl in LEVELS}


@pytest.fixture(scope="module")
def stratified() -> dict[str, object]:
    return {lvl.name: strata(lvl) for lvl in LEVELS}


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_delta_splits_into_a_draft_term_and_a_loop_term(decomposed, level):
    d = decomposed[level]
    delta, draft, loop = _EXPECTED[level]
    assert d.delta == pytest.approx(delta, abs=_TOL)
    assert d.draft_term == pytest.approx(draft, abs=_TOL)
    assert d.loop_term == pytest.approx(loop, abs=_TOL)
    # The split is an identity, not a model: it must close exactly.
    assert d.draft_term + d.loop_term == pytest.approx(d.delta, abs=1e-12)


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_offline_replay_reproduces_the_paid_runs_own_reported_delta(decomposed, level):
    """The decomposition is computed from replayed drafts plus released per-task scores.

    If it did not agree with the arm means the paid run itself reported, the
    decomposition would be describing some other quantity than the paper's headline.
    """
    lvl = next(x for x in LEVELS if x.name == level)
    assert decomposed[level].delta == pytest.approx(released_delta(lvl), abs=_TOL)


def test_the_sign_flip_survives_on_the_treated_component_alone(decomposed):
    """The registered result is a sign flip in Δ. It must not be an artifact of the
    shared `generate` gene, which both arms evolve and which is not the treatment."""
    assert decomposed["L1"].loop_term < 0
    assert decomposed["L2"].loop_term > 0
    # And it is *sharper* there: at L2 the shared gene contributes nothing at all.
    assert decomposed["L2"].draft_term == pytest.approx(0.0, abs=1e-12)


def test_at_L1_the_majority_of_the_loss_is_not_the_treatment(decomposed):
    """The headline -0.042 overstates the loop-level penalty: most of it is the draft."""
    d = decomposed["L1"]
    assert abs(d.draft_term) > abs(d.loop_term)
    assert abs(d.draft_term) / abs(d.delta) > 0.5


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_strata_are_exhaustive_and_close_on_the_pooled_delta(stratified, decomposed, level):
    s = stratified[level]
    assert sum(g.n for g in s.values()) == 30  # 3 seeds x 10 held-out
    assert sum(g.contribution for g in s.values()) == pytest.approx(
        decomposed[level].delta, abs=1e-12
    )


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_stratum_means_are_pinned(stratified, level):
    for correct, (n, mean_b, mean_d) in _STRATA[level].items():
        g = stratified[level][correct]
        assert g.n == n
        assert g.mean_b == pytest.approx(mean_b, abs=_STRATA_TOL)
        assert g.mean_d == pytest.approx(mean_d, abs=_STRATA_TOL)


def test_the_damage_channel_does_not_carry_the_L1_loss(stratified):
    """The paper's stated mechanism, tested directly and found not to hold.

    If a never-accepting critique lost by damaging drafts that needed no repair, the
    already-correct stratum would carry the deficit. It carries the opposite sign.
    """
    already_correct = stratified["L1"][True]
    assert already_correct.contribution > 0
    # And it could not carry -0.042 even in the worst case: per-task scores are in
    # [0, 1], so the channel's contribution is bounded by its weight.
    assert already_correct.n / 30 == pytest.approx(2 / 30, abs=1e-12)


def test_the_stratifier_is_pre_treatment(stratified):
    """Both arms are stratified on the SEED genome's first draft, so the covariate is
    identical across arms and conditioning on it cannot induce a collider bias. The
    guarantee we can assert here is that the strata partition the same task set for both
    arms, which an arm-specific stratifier would break."""
    for level in ("L1", "L2", "L3"):
        for g in stratified[level].values():
            assert g.n_b == g.n_d == g.n


#: level -> (accepted, refined, gain per refined) for the unoptimized seed genome.
_SEED_REFINE = {"L1": (7, 23, +0.072), "L2": (20, 10, +0.160), "L3": (18, 12, +0.042)}


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_the_refine_step_is_net_positive_with_no_optimizer(level):
    """Pins the numbers the paper uses to rule out 'the refine step is destructive'.

    If a refine round lost score on average, an arm whose critique never accepts would lose
    simply by invoking it 30 times. It does not: refine helps at every level.
    """
    lvl = next(x for x in LEVELS if x.name == level)
    acc, ref, gain = refine_productivity(lvl, SEED_ARM)
    want_acc, want_ref, want_gain = _SEED_REFINE[level]
    assert (acc, ref) == (want_acc, want_ref)
    assert gain == pytest.approx(want_gain, abs=_TOL)
    assert gain > 0


def test_the_loop_level_arm_refines_more_and_gains_less_per_refine_at_L1():
    """The mechanism the decomposition leaves standing for L1's loop term."""
    lvl = next(x for x in LEVELS if x.name == "L1")
    _, ref_b, gain_b = refine_productivity(lvl, "B")
    acc_d, ref_d, gain_d = refine_productivity(lvl, "D")
    assert acc_d == 0  # its critique never accepts
    assert ref_d == 30 and ref_b == 24
    assert gain_d == pytest.approx(0.043, abs=_TOL)
    assert gain_b == pytest.approx(0.075, abs=_TOL)
    assert gain_d < gain_b


#: level -> arm -> mean final-answer words. The do-nothing-floor threat, measured.
_FINAL_WORDS = {
    "L1": {"B": 26.4, "D": 27.3},
    "L2": {"B": 43.2, "D": 42.3},
    "L3": {"B": 46.1, "D": 45.5},
}


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_the_loop_level_arm_does_not_collapse_answer_length(level):
    """The metric is not anchored at zero for an empty answer, so a critique that pushed
    toward saying less would score better without answering better. Measured: it did not."""
    lvl = next(x for x in LEVELS if x.name == level)
    for arm, want in _FINAL_WORDS[level].items():
        _, final_words = answer_length(lvl, arm)
        assert final_words == pytest.approx(want, abs=0.05)
    # And the refine round moves length by under two words, against a floor that needs an
    # empty answer to reach.
    for arm in ("B", "D"):
        draft_words, final_words = answer_length(lvl, arm)
        assert abs(final_words - draft_words) < 2.0


# --------------------------------------------------------------------- the split-size sweep
#
# The sweep's own decomposition, pinned for the same reason the p0 levels are: these three
# rows cost real money, they are what withdrew the paper's negative headline, and a claim that
# lives only in prose is a claim that can drift. The rows also carry a property the p0 levels
# do not, which is why they are worth their own assertions rather than a wider parametrize:
# the loop term is positive at every bought size, so the treated component of Delta never
# reproduces the sign the paper originally reported.

#: level -> (delta, draft_term, loop_term), measured from the committed cassettes.
_SPLIT_EXPECTED = {
    "n=4": (+0.016667, +0.005000, +0.011667),
    "n=16": (+0.061667, +0.001667, +0.060000),
    "n=24": (+0.023333, -0.013333, +0.036667),
}

#: level -> stratum -> (n, mean_B, mean_D), on the seed genome's first-draft correctness.
_SPLIT_STRATA = {
    "n=4": {True: (4, 0.8125, 0.8125), False: (26, 0.7654, 0.7846)},
    "n=16": {True: (3, 0.7667, 0.7667), False: (27, 0.7463, 0.8148)},
    "n=24": {True: (7, 0.8000, 0.8714), False: (23, 0.7609, 0.7696)},
}

_SPLIT_NAMES = list(_SPLIT_EXPECTED)


@pytest.fixture(scope="module")
def split_decomposed() -> dict[str, object]:
    return {lvl.name: decompose(lvl) for lvl in SPLIT_LEVELS}


@pytest.mark.parametrize("level", _SPLIT_NAMES)
def test_split_decomposition_is_pinned(split_decomposed, level):
    d = split_decomposed[level]
    delta, draft, loop = _SPLIT_EXPECTED[level]
    assert d.delta == pytest.approx(delta, abs=_TOL)
    assert d.draft_term == pytest.approx(draft, abs=_TOL)
    assert d.loop_term == pytest.approx(loop, abs=_TOL)
    # The identity, restated per row: the two terms must exhaust Delta, not merely correlate
    # with it. A decomposition that leaves a residue is not a decomposition.
    assert d.draft_term + d.loop_term == pytest.approx(d.delta, abs=1e-9)


@pytest.mark.parametrize("level", _SPLIT_NAMES)
def test_split_offline_replay_reproduces_the_paid_runs_own_delta(split_decomposed, level):
    """The replay scores the released genomes itself. If it disagreed with the number the run
    reported, the decomposition would be describing a different experiment."""
    lvl = next(x for x in SPLIT_LEVELS if x.name == level)
    assert split_decomposed[level].delta == pytest.approx(released_delta(lvl), abs=_TOL)


@pytest.mark.parametrize("level", _SPLIT_NAMES)
def test_split_strata_are_pinned_and_exhaustive(level):
    lvl = next(x for x in SPLIT_LEVELS if x.name == level)
    s = strata(lvl)
    assert sum(v.n for v in s.values()) == 30
    for correct, (n, mean_b, mean_d) in _SPLIT_STRATA[level].items():
        assert s[correct].n == n
        assert s[correct].mean_b == pytest.approx(mean_b, abs=_STRATA_TOL)
        assert s[correct].mean_d == pytest.approx(mean_d, abs=_STRATA_TOL)


def test_the_treated_component_is_positive_at_every_bought_split_size(split_decomposed):
    """The load-bearing negative result of the paper is that the loop term is negative at
    n_train=8. It is positive at 4, 16 and 24, which is why the paper reports the -0.042 as one
    cell rather than as a finding about low p0. If this ever goes red, that withdrawal has to
    be revisited -- so it is asserted rather than left to the prose."""
    for level in _SPLIT_NAMES:
        assert split_decomposed[level].loop_term > 0, level
    l1_loop = decompose(next(x for x in LEVELS if x.name == "L1")).loop_term
    assert l1_loop < 0, "n_train=8 is the cell the sweep was built to interrogate"


def test_the_sweep_holds_the_seed_genome_fixed_while_its_score_moves(split_decomposed):
    """The sweep's rows score the same thirty held-out tasks with the same unoptimized genome,
    so their seed means are four draws of one number. They differ, and that difference is the
    measurement-noise floor the paper now reports. A test that they are EQUAL would be wrong;
    what is checkable is that they move by an amount consistent with that floor and not more.
    """
    import json

    seeds = []
    for slug in ("split4", "c2b", "split16", "split24"):
        runs = json.loads(pathlib.Path(f"docs/validation-study-{slug}.json").read_text())["runs"]
        seeds.append(sum(r["B"]["seed_heldout"] for r in runs) / len(runs))
    spread = max(seeds) - min(seeds)
    assert spread > 0, "identical means would mean the runs are not independent draws"
    # sigma(mean of 30) = 0.153/sqrt(30) = 0.0279; the range of four draws should not exceed a
    # few sigma. This is a smoke test on the noise story, not a hypothesis test.
    assert spread < 4 * 0.0279, f"seed-genome means span {spread:.4f}, too wide to be noise"
