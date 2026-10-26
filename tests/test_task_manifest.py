"""A frozen task manifest: levels built by measurement rather than by a generator knob.

Why this exists. Paper I's D9 names the open design problem: a p0 level needs a knob that
raises the base rate of already-correct first drafts WITHOUT raising the baseline, and
constraint exclusion cannot be that knob, because it re-assigns which constraints are
compulsory and so names a different generator rather than a difficulty. The candidate the
paper keeps is to measure p0 per task and *sample* tasks to a target. That needs the chosen
tasks frozen somewhere a replay can find them, which is this manifest.

The tests pin the properties a level has to have before money is spent on it: every task is
one the generator itself emitted (so the satisfiability guarantee carries over), no task is
scored in two places (so seeds stay independent draws), and a request for more tasks than a
level holds raises instead of quietly scoring fewer.
"""

from __future__ import annotations

import json

import pytest

from whystuck.eval.multi_constraint import mc_score
from whystuck.eval.task_manifest import (
    build_manifest,
    calibration_levels,
    calibration_pool,
    load_manifest,
    manifest_split,
    save_manifest,
    select_levels,
)


def _pool(n: int = 24):
    return calibration_pool(n, seed=1000)


def test_calibration_pool_is_hard_tier_with_unique_namespaced_ids():
    tasks = _pool(30)
    ids = [t.id for t in tasks]
    assert len(set(ids)) == 30
    assert all(i.startswith("p0cal-1000-") for i in ids)
    # Hard tier: 4-5 constraints, and exact_word_count is forced into every task.
    for t in tasks:
        cons = t.payload["instruction_id_list"]
        assert 4 <= len(cons) <= 5
        assert "exact_word_count" in cons


def test_calibration_pool_is_deterministic_and_seed_dependent():
    a, b, c = (
        calibration_pool(10, seed=1000),
        calibration_pool(10, seed=1000),
        calibration_pool(10, seed=1001),
    )
    assert [t.payload for t in a] == [t.payload for t in b]
    assert [t.payload for t in a] != [t.payload for t in c]


def test_round_trip_and_split(tmp_path):
    tasks = _pool(24)
    ids = [t.id for t in tasks]
    levels = {
        "low": {
            0: {"train": ids[0:4], "heldout": ids[4:9]},
            1: {"train": ids[9:13], "heldout": ids[13:18]},
        }
    }
    m = build_manifest({"n": 24, "seed": 1000}, tasks, levels, stats={"note": "x"})
    path = tmp_path / "m.json"
    save_manifest(path, m)
    loaded = load_manifest(path)
    got = manifest_split(loaded, "low", "heldout", seed=1, n=5)
    assert [t.id for t in got] == ids[13:18]
    assert [t.payload for t in got] == [t.payload for t in tasks[13:18]]
    assert manifest_split(loaded, "low", "train", seed=0, n=2) == tasks[0:2]


def test_split_refuses_more_tasks_than_the_level_holds(tmp_path):
    tasks = _pool(12)
    ids = [t.id for t in tasks]
    m = build_manifest(
        {"n": 12, "seed": 1000},
        tasks,
        {"x": {0: {"train": ids[:3], "heldout": ids[3:6]}}},
        stats={},
    )
    save_manifest(tmp_path / "m.json", m)
    loaded = load_manifest(tmp_path / "m.json")
    with pytest.raises(ValueError, match="only 3"):
        manifest_split(loaded, "x", "heldout", seed=0, n=4)


def test_split_refuses_unknown_level_and_seed(tmp_path):
    tasks = _pool(12)
    ids = [t.id for t in tasks]
    m = build_manifest(
        {"n": 12, "seed": 1000},
        tasks,
        {"x": {0: {"train": ids[:3], "heldout": ids[3:6]}}},
        stats={},
    )
    save_manifest(tmp_path / "m.json", m)
    loaded = load_manifest(tmp_path / "m.json")
    with pytest.raises(KeyError, match="level"):
        manifest_split(loaded, "nope", "heldout", seed=0, n=1)
    with pytest.raises(KeyError, match="seed"):
        manifest_split(loaded, "x", "heldout", seed=7, n=1)


def test_a_task_scored_twice_is_refused():
    """Overlap between seeds would make them re-partitions rather than independent draws,
    which is Paper III's defect 5. The manifest refuses it at build time."""
    tasks = _pool(12)
    ids = [t.id for t in tasks]
    levels = {
        "x": {
            0: {"train": ids[:3], "heldout": ids[3:6]},
            1: {"train": ids[5:8], "heldout": ids[8:11]},
        }
    }
    with pytest.raises(ValueError, match="more than once"):
        build_manifest({"n": 12, "seed": 1000}, tasks, levels, stats={})


def test_an_assignment_naming_a_missing_task_is_refused():
    tasks = _pool(6)
    with pytest.raises(ValueError, match="not in the manifest"):
        build_manifest(
            {"n": 6, "seed": 1000}, tasks, {"x": {0: {"train": [], "heldout": ["ghost"]}}}, stats={}
        )


def test_tampered_payload_is_refused_on_load(tmp_path):
    """A task edited by hand would no longer carry the generator's satisfiability proof."""
    tasks = _pool(6)
    ids = [t.id for t in tasks]
    m = build_manifest(
        {"n": 6, "seed": 1000}, tasks, {"x": {0: {"train": [], "heldout": ids[:2]}}}, stats={}
    )
    path = tmp_path / "m.json"
    save_manifest(path, m)
    raw = json.loads(path.read_text())
    raw["tasks"][0]["payload"]["goal"] += " extra"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        load_manifest(path)
    # Recomputing the digest does not launder it: the payload must match the generator.
    from whystuck.eval.task_manifest import _digest

    raw["sha256"] = _digest(raw["tasks"])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="generator"):
        load_manifest(path)


def test_calibration_levels_chunk_every_task_once():
    ids = [f"t{i}" for i in range(25)]
    lv = calibration_levels(ids, chunk=10)
    chunks = lv["calib"]
    assert sorted(chunks) == [0, 1, 2]
    assert [len(chunks[s]["heldout"]) for s in (0, 1, 2)] == [10, 10, 5]
    assert all(chunks[s]["train"] == [] for s in chunks)
    flat = [i for s in chunks for i in chunks[s]["heldout"]]
    assert flat == ids


# --- selection ---------------------------------------------------------------------------


def _stats(spec):
    """spec: list of (n, p_first, final) -> per-task stats dict."""
    out, k = {}, 0
    for n, p, f in spec:
        for _ in range(n):
            out[f"t{k}"] = {"p_first": p, "final": f}
            k += 1
    return out


def test_selection_hits_each_target_with_a_matched_baseline():
    # Targets are reachable means: with p_first in {0, 1}, a 10-task mean moves in steps of
    # 0.1, so a target between steps would be infeasible by granularity, not by design.
    # Three kinds of task: never right first time but repaired to 0.8; always right (1.0);
    # never right and stuck low (0.6). Mixing the last two raises p0 at a fixed mean final.
    stats = _stats([(200, 0.0, 0.8), (200, 1.0, 1.0), (200, 0.0, 0.6)])
    sel = select_levels(
        stats,
        targets={"low": 0.0, "mid": 0.3, "high": 0.5},
        seeds=[0, 1, 2],
        n_train=10,
        n_heldout=10,
        baseline=0.8,
        tol_p0=0.03,
        tol_baseline=0.03,
        rng_seed=7,
    )
    used = []
    for level, target in {"low": 0.0, "mid": 0.3, "high": 0.5}.items():
        for seed in (0, 1, 2):
            for split, n in (("train", 10), ("heldout", 10)):
                ids = sel["levels"][level][seed][split]
                assert len(ids) == n
                used += ids
            ho = sel["levels"][level][seed]["heldout"]
            p = sum(stats[i]["p_first"] for i in ho) / len(ho)
            f = sum(stats[i]["final"] for i in ho) / len(ho)
            assert abs(p - target) <= 0.03 + 1e-9
            assert abs(f - 0.8) <= 0.03 + 1e-9
    assert len(used) == len(set(used)), "selection reused a task"
    assert sel["achieved"]["high"][0]["heldout"]["p0"] == pytest.approx(0.5, abs=0.03)


def test_selection_is_deterministic():
    stats = _stats([(100, 0.0, 0.8), (100, 1.0, 1.0), (100, 0.0, 0.6)])
    kw = dict(
        targets={"a": 0.0, "b": 0.4},
        seeds=[0, 1],
        n_train=5,
        n_heldout=5,
        baseline=0.8,
        tol_p0=0.05,
        tol_baseline=0.05,
        rng_seed=3,
    )
    assert select_levels(stats, **kw) == select_levels(stats, **kw)


def test_selection_refuses_an_infeasible_level_instead_of_missing_quietly():
    # Every task with p_first > 0 is also final 1.0, and nothing is below 0.8, so a p0 of
    # 0.5 forces the mean final up to 0.9: the baseline cannot be held.
    stats = _stats([(200, 0.0, 0.8), (200, 1.0, 1.0)])
    with pytest.raises(ValueError, match="infeasible"):
        select_levels(
            stats,
            targets={"high": 0.5},
            seeds=[0],
            n_train=4,
            n_heldout=10,
            baseline=0.8,
            tol_p0=0.03,
            tol_baseline=0.03,
            rng_seed=0,
        )


def test_selection_refuses_when_the_pool_is_too_small():
    stats = _stats([(10, 0.0, 0.8)])
    with pytest.raises(ValueError, match="pool"):
        select_levels(
            stats,
            targets={"low": 0.0},
            seeds=[0, 1],
            n_train=4,
            n_heldout=5,
            baseline=0.8,
            tol_p0=0.05,
            tol_baseline=0.05,
            rng_seed=0,
        )


def test_selected_tasks_are_satisfiable_generator_output():
    tasks = _pool(40)
    for t in tasks:
        assert t.payload["instruction_id_list"]
    # The witness proof ran at generation; a manifest only ever carries generator output.
    from whystuck.eval.multi_constraint import _pairs, build_witness

    for t in tasks[:10]:
        assert mc_score(build_witness(_pairs(t)), t) == 1.0


def test_calibration_pool_can_hold_one_excluded_generator_fixed():
    """Deviation 2 of the target-p0 pre-registration: on qwen3.8 the hard tier has p0 = 0.019,
    so levels are sampled from ONE fixed generator with exact_word_count excluded. The pool
    must be exactly that generator's output, with ids that cannot collide with the hard tier."""
    from whystuck.eval.multi_constraint import generate_multi_tasks

    ex = frozenset({"exact_word_count"})
    tasks = calibration_pool(20, seed=1000, exclude=ex)
    ref = generate_multi_tasks(20, seed=1000, k_min=4, k_max=5, hard=True, exclude=ex)
    assert [t.payload for t in tasks] == [t.payload for t in ref]
    assert all(t.id.startswith("p0cal-1000-x-exact_word_count-") for t in tasks)
    assert all("exact_word_count" not in t.payload["instruction_id_list"] for t in tasks)
    assert calibration_pool(5, seed=1000)[0].id == "p0cal-1000-0"  # default ids unchanged


def test_manifest_round_trips_an_excluded_generator(tmp_path):
    ex = frozenset({"exact_word_count"})
    tasks = calibration_pool(8, seed=1000, exclude=ex)
    m = build_manifest(
        {"n": 8, "seed": 1000, "exclude": sorted(ex)},
        tasks,
        calibration_levels([t.id for t in tasks], chunk=4),
        stats={},
    )
    save_manifest(tmp_path / "m.json", m)
    loaded = load_manifest(tmp_path / "m.json")
    assert [t.payload for t in manifest_split(loaded, "calib", "heldout", seed=1, n=4)] == [
        t.payload for t in tasks[4:8]
    ]
    # The generator spec is part of the provenance: dropping `exclude` must fail the check.
    raw = json.loads((tmp_path / "m.json").read_text())
    raw["generator"].pop("exclude")
    (tmp_path / "m.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="generator"):
        load_manifest(tmp_path / "m.json")


def test_selection_can_also_match_a_covariate_such_as_generator_share():
    """With a pool drawn from two generators, p0-selection would otherwise also select for
    the generator, and the levels would differ in constraint mix as well as in p0. Matching
    the share removes that confound by construction."""
    # Both generators cover every (p_first, final) cell, so a 50/50 share is reachable at
    # every level; this test is about the matching, not about feasibility.
    cells = ((1.0, 1.0), (0.0, 0.8), (0.0, 0.6))
    spec = [(80, p, f, g) for p, f in cells for g in (0, 1)]
    stats, k = {}, 0
    for n, p, f, g in spec:
        for _ in range(n):
            stats[f"t{k}"] = {"p_first": p, "final": f, "l2": float(g)}
            k += 1
    sel = select_levels(
        stats, targets={"high": 0.5, "low": 0.1}, seeds=[0, 1], n_train=10, n_heldout=10,
        baseline=0.8, tol_p0=0.03, tol_baseline=0.03, rng_seed=1,
        covariates={"l2": (0.5, 0.05)},
    )  # fmt: skip
    for lv in ("high", "low"):
        for s in (0, 1):
            for split in ("train", "heldout"):
                ids = sel["levels"][lv][s][split]
                assert abs(sum(stats[i]["l2"] for i in ids) / len(ids) - 0.5) <= 0.05 + 1e-9
                assert sel["achieved"][lv][s][split]["l2"] == pytest.approx(
                    sum(stats[i]["l2"] for i in ids) / len(ids)
                )


def test_a_covariate_absent_from_the_stats_is_refused():
    stats = _stats([(50, 0.0, 0.8)])
    with pytest.raises(KeyError, match="l2"):
        select_levels(
            stats, targets={"a": 0.0}, seeds=[0], n_train=2, n_heldout=2, baseline=0.8,
            tol_p0=0.05, tol_baseline=0.05, rng_seed=0, covariates={"l2": (0.5, 0.1)},
        )  # fmt: skip


def test_a_manifest_can_mix_tasks_from_several_fixed_generators(tmp_path):
    ex = frozenset({"exact_word_count"})
    hard = calibration_pool(6, seed=1001)
    l2 = calibration_pool(6, seed=1001, exclude=ex)
    tasks = hard + l2
    gens = [{"n": 6, "seed": 1001}, {"n": 6, "seed": 1001, "exclude": sorted(ex)}]
    levels = {"mid": {0: {"train": [hard[0].id, l2[0].id], "heldout": [hard[1].id, l2[1].id]}}}
    m = build_manifest(gens, tasks, levels, stats={})
    save_manifest(tmp_path / "m.json", m)
    loaded = load_manifest(tmp_path / "m.json")
    got = manifest_split(loaded, "mid", "heldout", seed=0, n=2)
    assert [t.payload for t in got] == [hard[1].payload, l2[1].payload]
    # Drop the second generator from the provenance: its tasks no longer verify.
    raw = json.loads((tmp_path / "m.json").read_text())
    raw["generator"] = raw["generator"][:1]
    (tmp_path / "m.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="generator"):
        load_manifest(tmp_path / "m.json")
