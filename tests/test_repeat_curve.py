"""The curve replicates: the σ-scaled claims of Table `tab:curve` currently borrow a denominator.

WHY THIS EXPERIMENT EXISTS. Every ratio in the detector-quality curve divides an observed gap by
σ(Δ) = 0.0286, which was estimated from a *seed* genome at R=1 in a different run family. The
paper then measures, two tables later, that σ is a property of the genome and spans a factor of
5.9 (0.0057 to 0.0336). Three of the curve's ratios have a margin against the 2σ line smaller
than that spread, so they are neither confirmed nor refuted by the number they are scaled by.
The table below is parsed and checked against the artifacts by
`test_the_fragility_table_in_this_docstring_is_derived`, because a hand-typed margin is exactly
the kind of number this paper exists to complain about --- and a first draft of it did read the
tool lift off run 3's FIVE-seed mean (0.7650) where every gap uses the three-seed mean (0.7767),
which understated the fragility by a factor of five:

    run4 - run3    +0.0600   under 2 sigma if sigma >= 0.0300   (+5 % over the borrowed 0.0286)
    run10 - run9   +0.0633   under 2 sigma if sigma >= 0.0317   (+11 %)
    run8 - run7    +0.0650   under 2 sigma if sigma >= 0.0325   (+14 %)
    run7 - run4    -0.0883   under 2 sigma if sigma >= 0.0442   (+55 %)

Only the last, the mechanistic gap, sits outside the plausible envelope. This driver measures σ
for each curve row directly, the same way the 135 p₀ replicates did, so the ratios stop being
borrowed.

The failure modes are the ones that produce a clean table from a wrong experiment, so they are
pinned rather than reviewed: a row whose flags do not reproduce its released run measures the
noise of a configuration the paper never ran, and a pair differenced across replicates instead of
within one averages away the quantity being bought.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import repeat_scores as rs  # noqa: E402
import verify_paper_claims as vpc  # noqa: E402

CHECKER = pathlib.Path("scripts/verify_paper_claims.py")

#: Held-out tasks per seed. Every curve run used ten, so a pooled row is thirty tasks -- the same
#: scale as the p0 replicates, which is what makes the two sigma families comparable.
N_HELDOUT_PER_SEED = 10


def _meta(slug: str) -> dict:
    return json.loads(pathlib.Path(f"docs/validation-study-{slug}.json").read_text())["meta"]


# --------------------------------------------------------------------- configuration fidelity


@pytest.mark.parametrize("run", rs.CURVE, ids=lambda r: r.label)
def test_each_curve_row_reproduces_its_released_runs_configuration(run: rs.CurveRun):
    """The load-bearing test.

    A replicate that drops `--oracle-critique` measures the intrinsic critique's noise and files
    it under the oracle row; one that drops `--feedback-level hi` re-measures run 9 twice and
    reports the actionability gap as exactly zero variance. Both look like results. So the flag
    set is derived from the released run's own metadata rather than transcribed.
    """
    assert rs.curve_row_mismatches(run) == []


@pytest.mark.parametrize(
    "broken",
    [
        # The oracle rows without their gate: these would measure the intrinsic critique.
        rs.CurveRun("run7", "oracle", (), 1),
        rs.CurveRun("run8", "fixbudget", (), 3),
        # The actionability pair collapsed onto one level: the gap's variance would come back 0.
        rs.CurveRun("run10", "fbacthi", ("--oracle-critique", "--feedback-level", "lo"), 3),
        rs.CurveRun("run9", "fbactlo", ("--oracle-critique",), 3),
        # The refine budget silently reverted to the default.
        rs.CurveRun("run8", "fixbudget", ("--oracle-critique",), 1),
        # The tool row without its tool: this is the whole external-feedback claim.
        rs.CurveRun("run4", "feedback", (), 1),
    ],
    ids=lambda r: f"{r.label}-{'-'.join(r.flags) or 'noflags'}-R{r.retries}",
)
def test_a_wrong_flag_set_is_caught(broken: rs.CurveRun):
    """The detector has to fire on the rows it was built for, not merely pass on a clean table.

    Each entry above is a plausible transcription slip that would still produce six full columns
    of numbers, which is exactly why the check is derived from the released metadata.
    """
    assert rs.curve_row_mismatches(broken) != [], f"{broken} should not validate"


@pytest.mark.parametrize("run", rs.CURVE, ids=lambda r: r.label)
def test_each_curve_row_pins_the_task_generating_parameters(run: rs.CurveRun):
    """Tasks are a function of (task set, seed, n_train, n_heldout).

    The curve's whole licence for omitting a task-draw term is that its runs share held-out
    tasks. A replicate drawn at a different split size is scoring different tasks, and its gap
    would be a difficulty difference wearing a noise label.
    """
    meta = _meta(run.slug)
    argv = rs.curve_argv_for(rs.CurveJob(run, seed=1, k=0), execute=False)
    assert argv[argv.index("--n-train") + 1] == str(meta["n_train"])
    assert argv[argv.index("--n-heldout") + 1] == str(meta["n_heldout"])
    assert argv[argv.index("--task-set") + 1] == "multi-hard"
    assert argv[argv.index("--seeds") + 1] == "1"
    assert "--exclude-constraint" not in argv, "the curve runs excluded no constraints"


#: Metadata fields that change what a replicate scores. `curve_argv_for` sets some explicitly and
#: inherits the rest from CLI defaults, and an inherited default is exactly the kind of thing that
#: can move under a run without anyone passing a flag -- so these are compared against what the
#: released run RECORDED, not against what the argv says.
#:
#: Values are the CLI default each field takes when ABSENT. The older released runs omit the
#: levers they did not use, so a missing key means "the default" and not "different" -- comparing
#: raw `.get()` results reports False against None on every row, which a first draft of this test
#: duly did.
_SCORING_FIELDS = {
    "worker_model": "openai/gpt-4.1-mini",
    "worker_max_output_tokens": 320,
    "loop_max_retries": 1,
    "n_train": 16,
    "n_heldout": 16,
    "oracle_critique": False,
    "feedback_stats": False,
    "feedback_level": "mid",
    "plan_gene": False,
}


@pytest.mark.parametrize("run", rs.CURVE, ids=lambda r: r.label)
def test_a_bought_replicate_recorded_the_same_configuration_as_its_run(run: rs.CurveRun):
    """End-to-end fidelity, checked from the artifact rather than from the invocation.

    `curve_argv_for` passes the flags that differ between rows and lets the rest fall to CLI
    defaults --- worker model and the 320-token output cap among them. A default that drifted
    would produce replicates of a subtly different loop while every argv assertion still passed,
    and the measured σ would belong to a configuration the paper never ran. Comparing recorded
    metadata closes that gap. Skipped for rows not yet bought, so the check strengthens as the
    experiment completes instead of blocking on it.
    """
    job = rs.CurveJob(run, rs.CURVE_SEEDS[0], 0)
    if not job.done:
        pytest.skip(f"{run.label} not bought yet")
    got = json.loads(pathlib.Path(job.out).read_text())["meta"]
    want = _meta(run.slug)
    bad = [
        f"{f}: replicate {got.get(f, d)!r} vs run {want.get(f, d)!r}"
        for f, d in _SCORING_FIELDS.items()
        if got.get(f, d) != want.get(f, d)
    ]
    assert not bad, f"{run.label} replicate scored a different loop:\n  " + "\n  ".join(bad)


def test_the_curve_measures_every_row_the_paper_scales_by_sigma():
    """Derived from the checker, not from a transcribed list.

    `_CURVE_RUNS` is the set of runs whose gaps the paper reports in units of σ. Leaving one out
    would leave a published ratio resting on the borrowed denominator while its neighbours no
    longer do, which is worse than leaving all of them there.
    """
    assert {r.label for r in rs.CURVE} == set(vpc._CURVE_RUNS)
    assert {r.slug for r in rs.CURVE} == set(vpc._CURVE_RUNS.values())
    for run in rs.CURVE:
        assert vpc._CURVE_RUNS[run.label] == run.slug


def test_every_pair_the_checker_sigma_scales_is_aggregated():
    """The pairs, too, come from the checker's source rather than from this file's opinion."""
    src = CHECKER.read_text(encoding="utf-8")
    used = {
        tuple(sorted(m.groups())) for m in re.finditer(r'\(\s*"(run\d+)",\s*"(run\d+)"\s*\)', src)
    }
    assert used, "no curve pairs found in the checker; the pattern went stale"
    ours = {tuple(sorted((a, b))) for _, a, b in rs.CURVE_PAIRS}
    assert used <= ours, f"the paper σ-scales pairs this driver does not measure: {used - ours}"


def test_the_fragility_table_in_this_docstring_is_derived():
    """The motivation for a paid experiment must itself be checked against the artifacts.

    The first version of the table above was transcribed, and one row was wrong: it took run 3's
    five-seed mean (0.7650, which is what Table tab:curve prints) instead of the three-seed mean
    the gaps are actually computed on (0.7767, per `_curve_seed_mean`). That turned a 5 % margin
    into a 26 % one and moved the tool lift from the most fragile claim to the least. Exactly the
    failure this driver exists to fix, one level up.
    """
    rows = re.findall(
        r"^\s{4}(run\d+) - (run\d+)\s+([+-][\d.]+)\s+under 2 sigma if sigma >= ([\d.]+)",
        __doc__ or "",
        re.M,
    )
    assert len(rows) == 4, f"the docstring table went unparseable, found {len(rows)} rows"
    bad = []
    for b, a, gap_s, need_s in rows:
        gap, _ = vpc._curve_gap(a, b)
        if abs(gap - float(gap_s)) > 5e-5:
            bad.append(f"{b}-{a}: docstring says {gap_s}, artifacts give {gap:+.4f}")
        if abs(abs(gap) / 2 - float(need_s)) > 5e-5:
            bad.append(f"{b}-{a}: threshold {need_s} is not |{gap:+.4f}|/2")
    assert not bad, "the docstring's motivation disagrees with the artifacts:\n  " + "\n  ".join(
        bad
    )


def test_the_shared_seed_count_matches_the_checkers():
    """Run 3 has five seeds; seeds 3-4 draw different tasks. Pooling them would reintroduce the
    task-draw variance the pairing exists to remove."""
    assert tuple(range(vpc._CURVE_SHARED_SEEDS)) == rs.CURVE_SEEDS


# --------------------------------------------------------------------------------- mechanics


def test_curve_jobs_never_collide_with_the_135_already_bought():
    """A tag collision would either re-buy a completed replicate or overwrite a published one."""
    bought = {j.tag for j in rs.plan(k=5)}
    new = {j.tag for j in rs.curve_plan(k=5)}
    assert not (bought & new)
    assert len(new) == len(rs.CURVE) * len(rs.CURVE_SEEDS) * 5
    assert len({j.out for j in rs.curve_plan(k=5)}) == len(new)


def test_the_135_already_bought_are_still_recognised_as_done():
    """The guard on this change. If adding the curve moved any existing tag, `done` goes False
    for 135 completed jobs and the next `--yes` re-buys $4 of replicates."""
    jobs = rs.plan(k=5)
    assert len(jobs) == 135
    assert all(j.done for j in jobs), "an existing replicate stopped resolving to its output file"


def test_every_curve_job_carries_a_distinct_salt():
    """Without the salt, record-mode dedup hands K replicates one cached draw and the measured
    σ comes back 0 -- the most flattering artifact available to this paper."""
    jobs = [j for j in rs.curve_plan(k=5) if j.run.label == "run8" and j.seed == 0]
    argvs = [rs.curve_argv_for(j, execute=True) for j in jobs]
    assert [a[a.index("--sample") + 1] for a in argvs] == ["0", "1", "2", "3", "4"]


def test_curve_argv_scores_the_seed_genome_and_runs_no_optimizer():
    """Every curve row in the paper is an arm-A number: the unoptimized loop. A `--seed-gene`
    here would measure an evolved genome and label it the seed baseline."""
    argv = rs.curve_argv_for(rs.curve_plan(k=1)[0], execute=True)
    assert argv[argv.index("--arms") + 1] == "A"
    assert "--seed-gene" not in argv
    assert "--temperature" not in argv, "the released numbers were drawn at the provider default"
    assert "--record" in argv


def test_dry_run_is_the_default_and_buys_nothing():
    assert "--yes" not in rs.curve_argv_for(rs.curve_plan(k=1)[0], execute=False)


def test_curve_verify_argv_replays_and_cannot_spend():
    job = rs.curve_plan(k=1)[0]
    argv = rs.curve_verify_argv_for(job, out="/tmp/x.json")
    assert "--replay" in argv and "--yes" not in argv
    assert argv[argv.index("--out") + 1] == "/tmp/x.json", "must not overwrite the artifact"


# ------------------------------------------------------------------------------- aggregation


def test_curve_pairs_are_differenced_within_a_replicate():
    """Pooling replicates before differencing averages away the noise being measured.

    Checked on the arithmetic rather than the source: two replicates whose per-task vectors are
    each internally consistent must yield the per-replicate gaps, not the gap of the means.
    """
    a = [{"t1": 0.0, "t2": 1.0}, {"t1": 1.0, "t2": 0.0}]
    b = [{"t1": 0.5, "t2": 1.0}, {"t1": 1.0, "t2": 0.5}]
    gaps = rs.paired_gaps(a, b)
    assert gaps == pytest.approx([0.25, 0.25])


def test_paired_gaps_refuse_mismatched_task_sets():
    """A Δ across different task ids is not a Δ."""
    with pytest.raises(ValueError, match="task"):
        rs.paired_gaps([{"t1": 1.0}], [{"t2": 1.0}])


def test_sigma_by_run_and_aggregate_never_fabricate_a_spread_from_one_draw():
    """A single draw has no spread, and reporting $0$ for it would read as perfect precision.

    This replaced an assertion that both were — literally — *empty*, on the grounds that nothing
    had been bought yet. That was a claim about the calendar rather than about the code, and it
    duly failed on the pre-push hook of the commit that bought the replicates. The invariant
    worth pinning is that a row appears only once it has at least two complete replicates,
    whatever exists on disk.
    """
    # k=1 cannot support a spread however many replicates are on disk.
    assert rs.sigma_by_run(k=1) == {}
    assert rs.curve_aggregate(k=1) == []
    # Replicate indices beyond what was bought must be skipped, not counted as zeros.
    beyond = 500
    for label, sd in rs.sigma_by_run(k=beyond).items():
        assert sd >= 0.0, label
    for result in rs.curve_aggregate(k=beyond):
        assert result.k == len(result.replicate_deltas) >= 2, result.level
        assert result.n_tasks == len(rs.CURVE_SEEDS) * N_HELDOUT_PER_SEED


def test_the_fidelity_check_gates_the_spend_and_not_only_the_tests():
    """A test that only runs in CI does not stop a hand-typed `--curve --yes` at 2am."""
    import inspect

    src = inspect.getsource(rs._curve_run_jobs)
    assert "curve_row_mismatches" in src
    # Anchored on the call that actually spends, not on the signature -- a first draft compared
    # against the word "execute" and passed on any ordering, since it is a parameter name.
    assert src.index("curve_row_mismatches") < src.index("subprocess.run"), "must gate first"
    assert rs._curve_run_jobs([], execute=False) == 0, "a clean table must not be blocked"


def test_the_cost_estimate_comes_from_the_study_cli():
    """Not from a per-call rate multiplied by a guessed call count.

    The CLI already owns a token-level estimator, and it is the one that priced every run in the
    paper. A rate blended from R=1 jobs would underestimate the R=3 rows, whose calls are refine
    prompts carrying a previous draft.
    """
    import inspect

    src = inspect.getsource(rs.curve_cost_estimate)
    assert "Estimated total" in src, "the estimate must be parsed from the CLI's own output"
    assert '"--yes"' not in src, "a pricing pass must not be able to spend"
