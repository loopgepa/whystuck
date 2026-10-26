"""Frozen task manifests: p0 levels built by measurement instead of by a generator knob.

Paper I's D9 is the open design problem this answers. A p0 level needs a knob that raises the
base rate of already-correct first drafts without raising the baseline, and constraint
exclusion cannot supply one: it re-assigns which constraints are compulsory, so each level
names a different generator rather than a difficulty, and on a second worker the "ladder" came
out non-monotonic. The candidate the paper keeps is to measure p0 *per task* and sample tasks
to a target. That is a two-step procedure, and this module holds both halves:

1. ``calibration_pool`` + ``calibration_levels`` lay a large hard-tier pool out as arm-A-only
   "seeds", so the existing runner scores every task under the seed genome. Repeating with a
   different ``--sample`` salt buys independent draws, and per task we keep the share of first
   drafts that were already correct (``p_first``) and the mean final score (``final``).
2. ``select_levels`` picks disjoint train/held-out sets whose mean ``p_first`` hits each
   target while their mean ``final`` stays at one common baseline. Mixing tasks that are
   always right first time with tasks that stay wrong after repair raises p0 at a fixed
   baseline, which is exactly the separation exclusion could not make.

The chosen tasks are written to a manifest that a replay can find. Three refusals guard it,
each because the failure it prevents would otherwise look like a result: a task that is not
byte-identical to generator output (it would lose the satisfiability proof the generator runs
on every task), a task assigned twice (seeds would become re-partitions of one pool rather
than independent draws, Paper III's defect 5), and a request for more tasks than a level holds
(the run would score fewer tasks than it priced and reports).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.multi_constraint import generate_multi_tasks

MANIFEST_VERSION = 1
SPLITS = ("train", "heldout")

#: level -> seed -> split -> task ids
Levels = dict[str, dict[int, dict[str, list[str]]]]


def calibration_pool(
    n: int, *, seed: int, exclude: frozenset[str] = frozenset()
) -> list[BenchmarkTask]:
    """``n`` hard-tier tasks from the stock generator, ids namespaced by generator seed.

    The generator numbers tasks by index (``mc-{i}``), so two pools would share ids while
    holding different payloads. Namespacing keeps an id meaning one task everywhere.

    ``exclude`` selects ONE fixed generator with those constraint types removed (see
    ``generate_multi_tasks``). Levels are then sampled from that single generator, so selection
    is the only knob, which is the point: exclusion as a *level* names a different generator per
    level, exclusion as the *pool* names one generator for all of them."""
    tasks = generate_multi_tasks(n, seed=seed, k_min=4, k_max=5, hard=True, exclude=exclude)
    tag = "".join(f"x-{c}-" for c in sorted(exclude))
    return [
        BenchmarkTask(id=f"p0cal-{seed}-{tag}{i}", payload=t.payload) for i, t in enumerate(tasks)
    ]


def calibration_levels(ids: Sequence[str], *, chunk: int) -> Levels:
    """Lay ``ids`` out as one ``calib`` level whose "seeds" are consecutive held-out chunks.

    Arm A only scores held-out, so train stays empty and every task is scored exactly once
    per calibration run."""
    out: dict[int, dict[str, list[str]]] = {}
    for s, start in enumerate(range(0, len(ids), chunk)):
        out[s] = {"train": [], "heldout": list(ids[start : start + chunk])}
    return {"calib": out}


def _digest(tasks: list[dict[str, Any]]) -> str:
    blob = json.dumps(tasks, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _check_assignments(levels: Levels, known: set[str]) -> None:
    seen: dict[str, str] = {}
    for level, seeds in levels.items():
        for seed, splits in seeds.items():
            unknown_splits = set(splits) - set(SPLITS)
            if unknown_splits:
                raise ValueError(
                    f"level {level!r} seed {seed}: unknown split(s) {sorted(unknown_splits)}"
                )
            for split, ids in splits.items():
                where = f"{level}/{seed}/{split}"
                for tid in ids:
                    if tid not in known:
                        raise ValueError(f"{where}: task {tid!r} is not in the manifest")
                    if tid in seen:
                        raise ValueError(
                            f"task {tid!r} is assigned more than once ({seen[tid]} and {where}); "
                            "a task scored in two places makes those seeds re-partitions of one "
                            "pool rather than independent draws"
                        )
                    seen[tid] = where


def build_manifest(
    generator: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tasks: Sequence[BenchmarkTask],
    levels: Levels,
    *,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and assemble a manifest dict (see ``save_manifest``)."""
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids in manifest tasks")
    _check_assignments(levels, set(ids))
    rows = [{"id": t.id, "payload": t.payload} for t in tasks]
    return {
        "version": MANIFEST_VERSION,
        # One spec, or a list when the pool mixes fixed generators; load verifies every task
        # against the union of what the listed generators emit.
        "generator": (
            dict(generator) if isinstance(generator, Mapping) else [dict(g) for g in generator]
        ),
        "sha256": _digest(rows),
        "tasks": rows,
        "levels": {
            lv: {str(s): {sp: list(v) for sp, v in sd.items()} for s, sd in seeds.items()}
            for lv, seeds in levels.items()
        },
        "stats": dict(stats),
    }


def save_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TaskManifest:
    generator: dict[str, Any] | list[dict[str, Any]]
    tasks: dict[str, BenchmarkTask]
    levels: Levels
    sha256: str
    stats: dict[str, Any]


def load_manifest(path: str | Path) -> TaskManifest:
    """Load and re-verify a manifest: digest, generator provenance, disjointness."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != MANIFEST_VERSION:
        raise ValueError(f"manifest version {raw.get('version')!r}, expected {MANIFEST_VERSION}")
    rows = raw["tasks"]
    if _digest(rows) != raw["sha256"]:
        raise ValueError(f"{path}: task digest does not match; the manifest was edited")
    gen = raw["generator"]
    regenerated: dict[str, Any] = {}
    for spec in [gen] if isinstance(gen, dict) else gen:
        for t in calibration_pool(
            int(spec["n"]), seed=int(spec["seed"]), exclude=frozenset(spec.get("exclude", ()))
        ):
            regenerated[t.id] = t.payload
    for row in rows:
        if regenerated.get(row["id"]) != row["payload"]:
            raise ValueError(
                f"{path}: task {row['id']!r} is not byte-identical to the generator's output "
                f"for {gen}; it would not carry the generator's satisfiability proof"
            )
    tasks = {r["id"]: BenchmarkTask(id=r["id"], payload=r["payload"]) for r in rows}
    levels: Levels = {
        lv: {int(s): {sp: list(v) for sp, v in sd.items()} for s, sd in seeds.items()}
        for lv, seeds in raw["levels"].items()
    }
    _check_assignments(levels, set(tasks))
    return TaskManifest(gen, tasks, levels, raw["sha256"], raw.get("stats", {}))


def manifest_split(
    manifest: TaskManifest, level: str, split: str, *, seed: int, n: int
) -> list[BenchmarkTask]:
    """The first ``n`` tasks of ``level``/``seed``/``split``; refuses rather than truncates."""
    if level not in manifest.levels:
        raise KeyError(f"level {level!r} not in manifest (has {sorted(manifest.levels)})")
    seeds = manifest.levels[level]
    if seed not in seeds:
        raise KeyError(f"seed {seed} not in level {level!r} (has {sorted(seeds)})")
    ids = seeds[seed].get(split, [])
    if n > len(ids):
        raise ValueError(
            f"requested n={n} {split!r} tasks from {level!r}/seed {seed} but only {len(ids)} "
            "are assigned; scoring fewer would disagree with the priced and reported sample size"
        )
    return [manifest.tasks[i] for i in ids[:n]]


def _pick(
    pool: list[str],
    stats: Mapping[str, Mapping[str, float]],
    n: int,
    dims: Sequence[tuple[str, float, float]],
    rng: random.Random,
    *,
    restarts: int = 20,
    iters: int = 4000,
) -> list[str] | None:
    """Local search for ``n`` ids from ``pool`` whose mean on every (key, target, tol) in
    ``dims`` lands within tol of target."""
    if len(pool) < n:
        return None
    val = {key: {i: float(stats[i][key]) for i in pool} for key, _, _ in dims}

    def loss(sums: list[float]) -> float:
        return sum(((sm / n - t) / tol) ** 2 for sm, (_, t, tol) in zip(sums, dims, strict=True))

    def ok(sums: list[float]) -> bool:
        return all(
            abs(sm / n - t) <= tol + 1e-9 for sm, (_, t, tol) in zip(sums, dims, strict=True)
        )

    for _ in range(restarts):
        chosen = rng.sample(pool, n)
        chosen_set = set(chosen)
        sums = [sum(val[key][i] for i in chosen) for key, _, _ in dims]
        for _ in range(iters):
            if ok(sums):
                return sorted(chosen)
            k = rng.randrange(n)
            out = chosen[k]
            best, best_l, best_sums = None, loss(sums), sums
            for cand in rng.sample(pool, min(len(pool), 64)):
                if cand in chosen_set:
                    continue
                cs = [
                    sm - val[key][out] + val[key][cand]
                    for sm, (key, _, _) in zip(sums, dims, strict=True)
                ]
                cl = loss(cs)
                if cl < best_l:
                    best, best_l, best_sums = cand, cl, cs
            if best is not None:
                chosen_set.discard(out)
                chosen_set.add(best)
                chosen[k] = best
                sums = best_sums
        if ok(sums):
            return sorted(chosen)
    return None


def select_levels(
    stats: Mapping[str, Mapping[str, float]],
    *,
    targets: Mapping[str, float],
    seeds: Sequence[int],
    n_train: int,
    n_heldout: int,
    baseline: float,
    tol_p0: float,
    tol_baseline: float,
    rng_seed: int,
    covariates: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Disjoint train/held-out sets per level and seed, each matched on p0 AND baseline.

    Levels are filled in the order given, from one shared pool, and every split must land
    inside every tolerance. A level that cannot is reported as infeasible by raising: a level
    quietly built off-target would carry its target's name into the paper.

    ``covariates`` maps a further per-task stat to a (target, tol) every split must also hit,
    e.g. the share of tasks from one generator when the pool mixes two. Without it, selecting
    on p0 also selects on whatever correlates with p0, and the levels differ in that too."""
    covariates = dict(covariates or {})
    for key in covariates:
        missing = [i for i in stats if key not in stats[i]]
        if missing:
            raise KeyError(f"covariate {key!r} is absent from {len(missing)} task stats")
    need = len(targets) * len(seeds) * (n_train + n_heldout)
    if need > len(stats):
        raise ValueError(
            f"pool too small: {len(targets)} levels x {len(seeds)} seeds x "
            f"{n_train + n_heldout} tasks = {need} > {len(stats)} calibrated tasks"
        )
    rng = random.Random(rng_seed)
    remaining = sorted(stats)
    levels: Levels = {}
    achieved: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for level, target in targets.items():
        dims = [
            ("p_first", target, tol_p0),
            ("final", baseline, tol_baseline),
            *((key, t, tol) for key, (t, tol) in covariates.items()),
        ]
        levels[level], achieved[level] = {}, {}
        for seed in seeds:
            levels[level][seed], achieved[level][seed] = {}, {}
            for split, n in (("heldout", n_heldout), ("train", n_train)):
                if n == 0:
                    levels[level][seed][split] = []
                    continue
                got = _pick(remaining, stats, n, dims, rng)
                if got is None:
                    raise ValueError(
                        f"level {level!r} (target p0 {target}) is infeasible at seed {seed} "
                        f"{split}: no {n} remaining tasks have mean p_first within {tol_p0}, "
                        f"mean final within {tol_baseline} of baseline {baseline}"
                        + (f", and covariates {covariates}" if covariates else "")
                    )
                taken = set(got)
                remaining = [i for i in remaining if i not in taken]
                levels[level][seed][split] = got
                summary = {
                    "p0": sum(float(stats[i]["p_first"]) for i in got) / n,
                    "final": sum(float(stats[i]["final"]) for i in got) / n,
                }
                for key in covariates:
                    summary[key] = sum(float(stats[i][key]) for i in got) / n
                achieved[level][seed][split] = summary
    return {"levels": levels, "achieved": achieved}
