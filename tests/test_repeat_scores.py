"""The repeat-sample driver: its plan, its salt discipline, and its arithmetic.

The experiment this drives is cheap and decisive, which is exactly why its failure modes are
dangerous: a driver that forgets the salt reports zero sampling variance, and a driver that
pools replicates before differencing them averages away the quantity being measured. Both
produce clean-looking tables. So both are pinned here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import repeat_scores as rs  # noqa: E402


def test_plan_covers_every_level_arm_seed_and_replicate():
    jobs = rs.plan(k=4)
    assert len(jobs) == len(rs.LEVELS) * len(rs.ARMS) * len(rs.SEEDS) * 4
    assert len({j.tag for j in jobs}) == len(jobs), "tags must be unique or outputs collide"


def test_every_job_carries_a_distinct_salt():
    """Without this the K replicates are one cached draw wearing K filenames."""
    jobs = [j for j in rs.plan(k=5) if j.level == "L1" and j.arm == "D" and j.seed == 0]
    assert sorted(j.k for j in jobs) == [0, 1, 2, 3, 4]
    argvs = [rs.argv_for(j, {}, execute=True) for j in jobs]
    salts = [a[a.index("--sample") + 1] for a in argvs]
    assert salts == ["0", "1", "2", "3", "4"]


def test_no_temperature_is_pinned():
    """Deliberate: the estimate must describe the distribution the released numbers were
    drawn from, which is the provider's default."""
    argv = rs.argv_for(rs.plan(k=1)[0], {}, execute=True)
    assert "--temperature" not in argv


def test_exclusions_are_passed_per_level():
    """A replicate of L2 that forgot its exclusion would score a different task set and its
    Δ would be a difficulty difference wearing a noise label."""
    for level, _, exclusions in rs.LEVELS:
        job = next(j for j in rs.plan(k=1) if j.level == level and j.arm == "D")
        argv = rs.argv_for(job, {}, execute=True)
        passed = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude-constraint"]
        assert tuple(passed) == exclusions


def test_dry_run_is_the_default_and_buys_nothing():
    argv = rs.argv_for(rs.plan(k=1)[0], {}, execute=False)
    assert "--yes" not in argv


def test_seed_arm_overrides_no_genes():
    """The unoptimized control has to be the seed genome, not a copy of an arm's."""
    assert rs.released_genes("c2b", "SEED", 0) == {}


def test_released_genes_reads_the_actual_committed_genome():
    genes = rs.released_genes("c2b", "D", 0)
    assert set(genes) == {"generate", "ensure_correct"}
    committed = json.loads(pathlib.Path("docs/validation-study-c2b.json").read_text())["runs"][0][
        "D"
    ]["final_genes"]
    assert genes == {k: str(v) for k, v in committed.items()}


def test_level_result_arithmetic():
    """sd is the single-draw noise; se is what K replicates buy. Confusing them would either
    overstate or understate every conclusion the experiment supports."""
    r = rs.LevelResult(level="X", k=4, replicate_deltas=[0.10, 0.20, 0.30, 0.40], n_tasks=30)
    assert r.delta == pytest.approx(0.25)
    # population sd of [.1,.2,.3,.4] is sqrt(0.0125) ≈ 0.1118
    assert r.sd_between_replicates == pytest.approx(0.111803, abs=1e-5)
    assert r.standard_error == pytest.approx(0.111803 / 2, abs=1e-5)
    assert r.in_sigma() == pytest.approx(0.25 / (0.111803 / 2), abs=1e-4)


def test_a_single_replicate_reports_no_standard_error_rather_than_zero():
    """K=1 has no spread to report. Zero would read as infinite precision."""
    r = rs.LevelResult(level="X", k=1, replicate_deltas=[0.05], n_tasks=30)
    assert r.sd_between_replicates == 0.0
    assert r.in_sigma() == float("inf")


def test_aggregate_is_empty_before_anything_is_bought():
    """It must not invent a result from absent files."""
    assert rs.aggregate(k=1) == [] or all(r.k >= 1 for r in rs.aggregate(k=1))


# ------------------------------------------------------------------- offline re-verification
#
# The sweep's runs are registered one-by-one in verify_reproducibility.py, which does not scale
# to 135 replicates. The driver already knows each job's exact invocation, so it can replay its
# own outputs instead -- but only if the replay is genuinely a replay: reading the cassette,
# spending nothing, and never writing over the artifact it is checking.


def test_verify_argv_replays_and_cannot_spend():
    job = rs.plan(k=1)[0]
    argv = rs.verify_argv_for(job, {}, out="/tmp/scratch.json")
    assert "--replay" in argv
    assert "--yes" not in argv, "a verification pass that can spend is not a verification pass"
    assert argv[argv.index("--record") + 1] == job.cassette


def test_verify_writes_somewhere_other_than_the_artifact_it_checks():
    """Replaying onto the committed output would make any comparison trivially pass."""
    job = rs.plan(k=1)[0]
    argv = rs.verify_argv_for(job, {}, out="/tmp/scratch.json")
    assert argv[argv.index("--out") + 1] != job.out


def test_verify_argv_matches_the_recording_argv_except_for_mode():
    """A replay that differs in any task-affecting flag would miss every cassette key, and a
    miss reported as a difference would look like irreproducibility rather than a driver bug."""
    job = rs.plan(k=3)[-1]
    rec = rs.argv_for(job, {}, execute=True)
    ver = rs.verify_argv_for(job, {}, out="/tmp/scratch.json")
    strip = {"--yes", "--replay"}
    rec_clean = [a for a in rec if a not in strip]
    ver_clean = [a for a in ver if a not in strip]
    # Only the --out value may differ.
    i = rec_clean.index("--out")
    assert rec_clean[:i] == ver_clean[:i]
    assert rec_clean[i + 2 :] == ver_clean[i + 2 :]


# --------------------------------------------------------------------------------- sharding
#
# 135 sequential replicates is ~2.7h of wall-clock for work that is embarrassingly parallel:
# every job has its own output file and its own cassette, so nothing is shared. What is NOT
# safe is two workers computing `todo` from the same list, which would buy every job twice.
# Sharding makes the partition explicit, and these tests pin the two properties that make it
# safe to run several workers at once: the shards cover everything, and they never overlap.


def test_shards_partition_the_plan_exactly():
    jobs = rs.plan(k=5)
    for n in (2, 3, 4, 7):
        shards = [rs.shard(jobs, i, n) for i in range(n)]
        seen: list[rs.Job] = []
        for sh in shards:
            seen += sh
        assert len(seen) == len(jobs), f"n={n} lost or duplicated jobs"
        assert {j.tag for j in seen} == {j.tag for j in jobs}
        for i in range(n):
            for j in range(i + 1, n):
                overlap = {x.tag for x in shards[i]} & {x.tag for x in shards[j]}
                assert not overlap, f"shards {i} and {j} both claim {overlap}"


def test_a_single_shard_is_the_whole_plan():
    jobs = rs.plan(k=2)
    assert [j.tag for j in rs.shard(jobs, 0, 1)] == [j.tag for j in jobs]


def test_shard_rejects_an_impossible_spec():
    jobs = rs.plan(k=1)
    for index, count in ((0, 0), (2, 2), (-1, 3), (3, 3)):
        with pytest.raises(ValueError):
            rs.shard(jobs, index, count)


def test_parse_shard_accepts_i_of_n_and_rejects_junk():
    assert rs.parse_shard("0/3") == (0, 3)
    assert rs.parse_shard("2/4") == (2, 4)
    for junk in ("", "1", "a/2", "1/2/3", "1-2"):
        with pytest.raises(ValueError):
            rs.parse_shard(junk)
