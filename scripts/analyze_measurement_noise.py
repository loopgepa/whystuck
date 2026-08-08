#!/usr/bin/env python
"""Recover the variance component the study never measured (zero cost, offline).

Every interval in this paper describes task-draw and search-path variation and explicitly not
measurement noise: the worker runs at the provider's default temperature, one sample per task,
so nothing was measured twice. Stated as a scope note, that is accurate. Left there, it invites
the wrong inference -- that noise is not a concern -- when what it actually means is that every
interval OMITS a component whose size was unknown.

It is knowable for free. Several released runs evaluate the **same unoptimized seed genome** on
the **same held-out tasks** before optimizing anything: the four-arm re-run and each row of the
split-size sweep all do. Those are independent draws of one fixed quantity, so the spread
between them is the missing component.

    sd(paired difference) = sigma * sqrt(2)        two independent draws of one task
    sigma                 = sd(diff) / sqrt(2)     one draw of one task
    sigma(Delta per task) = sigma * sqrt(2)        D and B are separate draws too
    sigma(Delta of mean)  = sigma * sqrt(2) / sqrt(n_tasks)

The last line is the number that qualifies every Delta in the paper.

THE HAZARD THIS MODULE IS BUILT AROUND. Task ids are generated positionally, so two runs whose
constraint pools differ carry the *same* ids: `c2b` and the p0 levels have byte-identical
held-out id sets while scoring genuinely different tasks. Pooling them would book a difficulty
difference (seed means 0.788 vs 0.855) as sampling noise and roughly double the estimate. An
id-equality check cannot see it. So eligibility is decided by the recorded configuration --
exclusions and loop levers -- and identity is then verified on task *payloads*, never on ids.

    python scripts/analyze_measurement_noise.py
"""

from __future__ import annotations

import glob
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

DOCS = "docs"

#: Levers that change what the seed loop does, so a run carrying one is not a re-measurement
#: of the same quantity. Each is a meta flag written by `run_validation_study`.
_LEVERS = ("plan_gene", "feedback_stats", "oracle_critique", "actuator", "exec_feedback")

#: The regime the replicates must share. Anything else is a different measurement.
_TASK_SET = "multi-hard"
_N_HELDOUT = 10
_N_SEEDS = 3

#: The worker whose genome this sigma describes. Part of the regime, and it was MISSING from
#: eligibility until 2026-08-08: noise is measured by re-scoring one fixed seed genome, so a run
#: executed by a different model measures a different quantity. A three-seed `llama-3.1-8b` study
#: added to `docs/` for an unrelated experiment therefore enrolled as an eleventh replicate and
#: shifted every published sigma at once. Exported so a test can assert the screen exists.
WORKER_MODEL = "openai/gpt-4.1-mini"

#: Routing the critique or the verdict to another model changes the seed loop, so such a run is
#: not a re-measurement either. Checked separately from `_LEVERS` because these carry a model
#: name rather than a boolean.
_ROUTING_KEYS = ("critique_model", "verdict_model")


class IncompatibleReplicates(RuntimeError):
    """Raised rather than pooling runs that do not measure the same thing."""


@dataclass(frozen=True)
class Replicate:
    name: str
    path: str
    exclusions: frozenset[str]
    levers: tuple[str, ...]
    gene_overrides: tuple[str, ...]


@dataclass(frozen=True)
class Noise:
    n_replicates: int
    n_tasks: int
    replicate_means: dict[str, float]
    mean_signed: float
    sd_paired_difference: float

    @property
    def sigma_single(self) -> float:
        """Sd of one task's score under one draw of a fixed genome."""
        return self.sd_paired_difference / 2**0.5

    @property
    def sigma_delta_task(self) -> float:
        """Sd of Δ(D−B) on a single task: two independent draws, so sigma*sqrt(2)."""
        return self.sigma_single * 2**0.5

    @property
    def sigma_delta_mean(self) -> float:
        """Sd of Δ(D−B) as the paper reports it -- a mean over `n_tasks`."""
        return self.sigma_delta_task / self.n_tasks**0.5

    def in_sigma(self, delta: float) -> float:
        return abs(delta) / self.sigma_delta_mean


def _doc(name: str) -> dict:
    return json.loads(Path(f"{DOCS}/validation-study-{name}.json").read_text())


def _name_of(path: str) -> str:
    stem = Path(path).stem.replace("validation-study", "").lstrip("-")
    return stem or "run1"


def eligible_replicates() -> list[Replicate]:
    """Released runs that re-measure the same seed genome on the same regime.

    Eligibility is read off the recorded configuration, never inferred from the data: a run
    with an exclusion, a lever, or a hand-written gene is measuring something else, and the
    scores alone cannot tell you which.
    """
    out: list[Replicate] = []
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        doc = json.loads(Path(path).read_text())
        meta, runs = doc["meta"], doc["runs"]
        if meta.get("task_set") != _TASK_SET or meta.get("n_heldout") != _N_HELDOUT:
            continue
        if meta.get("worker_model") != WORKER_MODEL:
            continue
        if any(meta.get(k) for k in _ROUTING_KEYS):
            continue
        if len(runs) != _N_SEEDS:
            continue
        arm = next((a for a in ("A", "B", "C", "D") if isinstance(runs[0].get(a), dict)), None)
        if arm is None or runs[0][arm].get("seed_heldout_by_task") is None:
            continue
        exclusions = frozenset(meta.get("exclude_constraint") or ())
        levers = tuple(k for k in _LEVERS if meta.get(k))
        overrides = tuple(sorted(meta.get("gene_overrides") or ()))
        if exclusions or levers or overrides:
            continue
        out.append(Replicate(_name_of(path), path, exclusions, levers, overrides))
    return out


def heldout_ids(name: str) -> list[list[str]]:
    """Per seed, the held-out task ids a run scored. Exposed because their COLLISION across
    incompatible runs is the hazard, so a test has to be able to assert it."""
    doc = _doc(name)
    out = []
    for run in doc["runs"]:
        arm = next(a for a in ("A", "B", "C", "D") if isinstance(run.get(a), dict))
        out.append(sorted(run[arm]["seed_heldout_by_task"]))
    return out


def _payload_signatures(name: str) -> list[list[str]]:
    """Per seed, a canonical signature of each held-out task's CONTENT.

    Ids cannot establish identity; payloads can. Regenerating them from the loader is
    legitimate here because the loader is deterministic given (seed, pool, tier, exclusions),
    which is precisely the tuple eligibility already fixed.
    """
    from whystuck.eval.multi_constraint import load_multi_constraint

    doc = _doc(name)
    exclusions = frozenset(doc["meta"].get("exclude_constraint") or ())
    pool = max(64, 2 * _N_HELDOUT + 2)
    out = []
    for seed in range(len(doc["runs"])):
        tasks = load_multi_constraint(
            "heldout", seed=seed, n=_N_HELDOUT, pool=pool, hard=True, exclude=exclusions
        )
        out.append(
            [
                json.dumps(
                    [
                        t.payload.get("instruction_id_list"),
                        t.payload.get("kwargs"),
                        t.payload.get("goal"),
                    ],
                    sort_keys=True,
                )
                for t in tasks
            ]
        )
    return out


def seed_vectors(names: list[str]) -> dict[str, list[float]]:
    """Per run, the seed genome's per-task held-out scores, flattened across seeds.

    Refuses to pool runs that are not measuring the same thing, in the order a mistake is
    most likely: configuration first (an exclusion is invisible in the scores), then payload
    identity, then id alignment.
    """
    if len(names) < 2:
        raise IncompatibleReplicates(
            f"need at least two replicates to estimate a spread, got {names}"
        )
    configs = {}
    for name in names:
        meta = _doc(name)["meta"]
        configs[name] = (
            frozenset(meta.get("exclude_constraint") or ()),
            tuple(k for k in _LEVERS if meta.get(k)),
            tuple(sorted(meta.get("gene_overrides") or ())),
        )
    first_name, first_cfg = next(iter(configs.items()))
    for name, cfg in configs.items():
        if cfg != first_cfg:
            raise IncompatibleReplicates(
                f"{name} and {first_name} differ in exclusion/lever/override configuration "
                f"({cfg} vs {first_cfg}). Their held-out task IDS may still match exactly -- "
                "ids are positional -- so pooling them would book a difficulty difference as "
                "sampling noise. Refusing."
            )
    sigs = {name: _payload_signatures(name) for name in names}
    for name in names[1:]:
        if sigs[name] != sigs[names[0]]:
            raise IncompatibleReplicates(
                f"{name} and {names[0]} scored different held-out task payloads despite a "
                "matching configuration; the loader or the pool rule has changed"
            )

    out: dict[str, list[float]] = {}
    reference: list[str] | None = None
    for name in names:
        doc = _doc(name)
        flat: list[float] = []
        ids: list[str] = []
        for run in doc["runs"]:
            arm = next(a for a in ("A", "B", "C", "D") if isinstance(run.get(a), dict))
            by_task = run[arm]["seed_heldout_by_task"]
            for key in sorted(by_task):
                ids.append(key)
                flat.append(by_task[key])
        if reference is None:
            reference = ids
        elif ids != reference:
            raise IncompatibleReplicates(f"{name}'s held-out id order differs from {names[0]}'s")
        out[name] = flat
    return out


def noise_estimate(names: list[str] | None = None) -> Noise:
    """The missing variance component, from every compatible replicate available."""
    chosen = names if names is not None else [r.name for r in eligible_replicates()]
    vectors = seed_vectors(chosen)
    sds, signed = [], []
    for a, b in itertools.combinations(chosen, 2):
        diff = [y - x for x, y in zip(vectors[a], vectors[b], strict=True)]
        # Sample sd (n−1). The population sd was used until 2026-07-31; over 30 paired tasks it
        # understates σ by √(30/29) = 1.017, which moved σ(Δ) from 0.0291 to 0.0286 and inflated
        # every ratio scaled by it by 1.7%. No verdict turned on it -- the bias is a quarter of
        # this estimate's own acknowledged drift confound -- but it ran in the flattering
        # direction, and the same mistake at K=5 (see `repeat_scores.sd_between_replicates`) did
        # move a verdict, so both are corrected rather than the one that mattered.
        sds.append(stdev(diff))
        signed.append(mean(diff))
    return Noise(
        n_replicates=len(chosen),
        n_tasks=len(vectors[chosen[0]]),
        replicate_means={k: mean(v) for k, v in vectors.items()},
        mean_signed=mean(signed),
        sd_paired_difference=mean(sds),
    )


def main() -> int:  # pragma: no cover - CLI convenience
    reps = eligible_replicates()
    est_means = seed_vectors([r.name for r in reps])
    print("=== the variance component the study's intervals omit ===")
    print("Independent re-measurements of ONE fixed seed genome on ONE fixed held-out set.\n")
    print(f"{'replicate':12} {'n_train':>7}  seed-genome mean over the same tasks")
    for r in reps:
        n_train = _doc(r.name)["meta"].get("n_train")
        own = mean(est_means[r.name])
        print(f"{r.name:12} {str(n_train):>7}  {own:.4f}")

    est = noise_estimate()
    print(f"\nreplicates                    {est.n_replicates}")
    print(f"tasks per replicate           {est.n_tasks}")
    print(f"mean signed difference        {est.mean_signed:+.4f}  (a re-measurement: ~0 expected)")
    print(f"sd of a paired difference     {est.sd_paired_difference:.4f}")
    print(f"sigma, one task one draw      {est.sigma_single:.4f}")
    print(f"sigma of Delta on one task    {est.sigma_delta_task:.4f}")
    print(
        f"sigma of Delta over {est.n_tasks} tasks  {est.sigma_delta_mean:.4f}"
        "   <-- qualifies every Delta"
    )

    print("\n=== the paper's headline effects, in units of that noise ===")
    for label, delta in (
        ("L1  four-arm re-run", -0.0417),
        ("L2  p0 sign flip", 0.0600),
        ("L3  p0 saturated", -0.0017),
        ("actionability hi-lo", 0.0630),
    ):
        print(f"  {label:22} Δ={delta:+.4f}  {est.in_sigma(delta):.2f} sigma")
    print(
        "\nRead this as scope, not as a refutation: the mean signed difference is confounded\n"
        "with any provider-side drift between recording dates, and three seeds cannot separate\n"
        "them. What it establishes is an order of magnitude, and that order of magnitude is the\n"
        "same as the effects being reported."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
