#!/usr/bin/env python
"""Compute the study's headline effect sizes + intervals from the committed results.

Zero cost, offline: reads the released per-seed results JSONs (no cassette replay,
no model) and derives the two load-bearing statistics the paper reports, so the
article's numbers are backed by a reproducible computation rather than hand math:

  1. the null  Δ(D−B) = mean_seed(D.opt_heldout − B.opt_heldout), per fair run;
  2. the one positive effect Δ(hi−lo) = mean_seed(hi.A.seed_heldout − lo...),

each with an EXACT-resample 95% interval (all n**n resamples of the paired
per-seed differences — exact at the small n here, no RNG, fully deterministic).

    python scripts/analyze_deltas.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import mean

from whystuck.eval.benchmark import seeds_are_independent

DOCS = "docs"

#: meta["dataset"] prefix -> task set, for runs recorded before meta["task_set"] existed.
#: Prefix-matched (the strings carry parenthetical detail), longest first so
#: "multi-constraint-hard" cannot be swallowed by "multi-constraint".
_DATASET_TO_TASK_SET = {
    "multi-constraint-hard": "multi-hard",
    "multi-constraint": "multi",
    "BigCodeBench-Hard": "bcb",
    "MBPP+": "code",
    "IFBench": "ifbench",
}


def _runs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _meta(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["meta"]


def task_set_of(meta: dict) -> str:
    """The task set a released run used, from its metadata.

    Prefers `meta["task_set"]`; the earliest runs predate that key, so it falls back to
    the `dataset` string, which every released run carries. Raises on anything it cannot
    place: the answer decides whether a cross-seed statistic may be read as replication,
    and guessing would hand a new task set the permissive answer by default.
    """
    explicit = meta.get("task_set")
    if explicit:
        return str(explicit)
    dataset = str(meta.get("dataset", ""))
    for prefix, task_set in sorted(_DATASET_TO_TASK_SET.items(), key=lambda kv: -len(kv[0])):
        if dataset.startswith(prefix):
            return task_set
    raise ValueError(
        f"cannot determine the task set for a run whose dataset is {dataset!r} and which has "
        "no meta['task_set']; add it to _DATASET_TO_TASK_SET after checking whether its "
        "loader regenerates the task pool per seed"
    )


def heldout_overlap(task_set: str, seeds: list[int], n_heldout: int | None = None) -> float:
    """Mean pairwise fraction of held-out TASKS shared between seeds, measured.

    Compares task CONTENT, not ids: the multi_constraint generator numbers tasks by index,
    so two seeds reuse the same ids for entirely different tasks (measured: 15-17 of 32 ids
    shared, 0 payloads shared). Keying on ids would report those runs as ~50% overlapping
    when they in fact share nothing, i.e. exactly backwards.
    """
    from whystuck.eval.bcb_bench import load_bcb_tasks
    from whystuck.eval.code_bench import load_code_tasks
    from whystuck.eval.ifbench import load_ifbench
    from whystuck.eval.multi_constraint import load_multi_constraint

    def load(seed: int) -> set[str]:
        if task_set in ("multi", "multi-hard"):
            pool = max(64, 2 * (n_heldout or 32) + 2)
            tasks = load_multi_constraint(
                "heldout", seed=seed, n=n_heldout, pool=pool, hard=task_set == "multi-hard"
            )
        elif task_set == "bcb":
            tasks = load_bcb_tasks("heldout", seed=seed, n=n_heldout)
        elif task_set == "code":
            tasks = load_code_tasks("heldout", seed=seed, n=n_heldout)
        else:
            tasks = load_ifbench(split="heldout", seed=seed, n=n_heldout)
        return {json.dumps(t.payload, sort_keys=True, default=str) for t in tasks}

    splits = {s: load(s) for s in seeds}
    sizes = [len(v) for v in splits.values()]
    if len(seeds) < 2 or not all(sizes):
        return 0.0
    fracs = [
        len(splits[a] & splits[b]) / min(len(splits[a]), len(splits[b]))
        for i, a in enumerate(seeds)
        for b in seeds[i + 1 :]
    ]
    return mean(fracs)


def delta_arm(
    results_json: str, a: str = "D", b: str = "B", field: str = "opt_heldout"
) -> list[float]:
    """Per-seed paired difference arm_a − arm_b on `field` (default D−B on opt held-out)."""
    return [r[a][field] - r[b][field] for r in _runs(results_json)]


def delta_between(
    json_hi: str, json_lo: str, arm: str = "A", field: str = "seed_heldout"
) -> list[float]:
    """Per-seed paired difference between two runs (same seeds/tasks), e.g. hi−lo on the
    seed held-out. Requires equal seed counts (paired by seed index)."""
    hi, lo = _runs(json_hi), _runs(json_lo)
    if len(hi) != len(lo):
        raise ValueError(f"seed-count mismatch: {json_hi} has {len(hi)}, {json_lo} has {len(lo)}")
    return [h[arm][field] - lo[i][arm][field] for i, h in enumerate(hi)]


def exact_resample_ci(diffs: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) with an EXACT percentile bootstrap: enumerate all n**n resamples
    (deterministic, no RNG) and take the alpha/2 and 1−alpha/2 percentile of the resample
    means. Exact at the small n used here (5**5 = 3125, 3**3 = 27).

    Indexing is `ceil(q * N) - 1`, the smallest index k with P(X <= means[k]) >= q. This
    matters because the resample distribution is DISCRETE: at n = 3 each of the 27 atoms
    carries 1/27 = 3.70% > alpha/2, so no single atom may be dropped from either tail.
    A previous implementation used `int(q * N) - 1` for the upper bound, which discarded
    the largest atom and reported an interval too narrow on that side — narrow in exactly
    the direction that makes an effect bracketing zero look like it excludes zero. The
    invariant the tests pin is per-tail: P(X < lo) <= alpha/2 and P(X > hi) <= alpha/2.
    """
    if not diffs:
        raise ValueError("no differences given")
    n = len(diffs)
    means = sorted(mean(s) for s in itertools.product(diffs, repeat=n))
    total = len(means)
    lo = means[math.ceil((alpha / 2) * total) - 1]
    hi = means[math.ceil((1 - alpha / 2) * total) - 1]
    return mean(diffs), lo, hi


def sign_test_p(diffs: list[float]) -> float:
    """Two-sided exact sign-test p-value on the paired per-seed differences.

    Ties (exactly 0.0) are dropped, per the standard sign test: a seed on which the two
    arms optimized to byte-identical genomes contributes no signed information — the
    cassette serves both arms the same completion, so the 0.0 is an identity, not an
    observation. Reported alongside the interval because at these seed counts the sign
    test is the honest floor: n = 3 unanimous gives p = 0.25 at best, so no result here
    can reach conventional significance on sign alone, and any "excludes zero" claim
    rests on the magnitudes rather than on replication across seeds.
    """
    signed = [d for d in diffs if d != 0.0]
    n = len(signed)
    if n == 0:
        return 1.0
    k = sum(1 for d in signed if d > 0)
    # Two-sided: double the smaller tail of Binomial(n, 0.5), clamped at 1.0.
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    sf = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * min(cdf, sf))


# (label, per-seed differences) for the fair runs' D−B null. Includes the cross-domain
# BigCodeBench-Hard run (D5); MBPP+ is excluded because it saturated its training split
# (all arms == seed, a degenerate zero — see the paper appendix), not a fair test.
#
# Run 10 (fbacthi) was previously listed here and has been REMOVED: it sets
# `oracle_critique`, under which the verify gene is inert by construction
# (`llm_worker.py`: "Not a D-vs-B test — the verify gene cannot matter under a perfect
# oracle"). Its Δ(D−B) = 0 is an identity, not a measurement, so citing its [0,0] as
# evidence that the null is tight was circular. The same applies to any oracle run.
#: The pre-registered budget sweep at fixed n_train=8 (D4). Effective search depth is
#: min(promotable cycles, --reflection-calls), so the label carries the depth rather than the
#: budget: at a cap of 6, mmc 80 and mmc 160 would buy the same six proposals.
_BUDGET_RUNS = [
    ("budget mmc=20  (depth 1.09)", f"{DOCS}/validation-study-mmc20.json"),
    ("budget mmc=28  (depth 1.82)", f"{DOCS}/validation-study-c2b.json"),
    ("budget mmc=40  (depth 2.91)", f"{DOCS}/validation-study-mmc40.json"),
    ("budget mmc=56  (depth 4.36)", f"{DOCS}/validation-study-mmc56.json"),
    ("budget mmc=80  (depth 6.00)", f"{DOCS}/validation-study-mmc80.json"),
    ("budget mmc=160 (depth 12.0)", f"{DOCS}/validation-study-mmc160.json"),
]

_NULL_RUNS = [
    # The p0 sweep leads, because it is the only place in this list where the sign of Δ
    # changes and therefore the only place a Δ carries a direction rather than an absence.
    # L1 IS the C2b re-run; the three rows are one axis, not three experiments, and reading
    # any of them alone reproduces the mistake the sweep exists to correct.
    ("p0 L1 p0=0.067 (n=3)", f"{DOCS}/validation-study-c2b.json"),
    ("p0 L2 p0=0.267 (n=3)", f"{DOCS}/validation-study-p0-L2.json"),
    ("p0 L3 p0=0.667 SATURATED", f"{DOCS}/validation-study-p0-L3.json"),
    # The split-size sweep, second, because it tests the one alternative explanation for the
    # L1 row above: that eight training tasks is too small a split for the two-gene arm. Read
    # as an axis like the p0 rows, not as three runs. n_train=8 is the p0 L1 row.
    ("split n_train=4  (n=3)", f"{DOCS}/validation-study-split4.json"),
    ("split n_train=16 (n=3)", f"{DOCS}/validation-study-split16.json"),
    ("split n_train=24 (n=3)", f"{DOCS}/validation-study-split24.json"),
    # Below: runs whose arms B and D emitted byte-identical genomes, so their Δ = 0 records
    # the absence of a treatment rather than the absence of an effect. Not interchangeable
    # with the rows above.
    ("C2    [contaminated seed2]", f"{DOCS}/validation-study-c2.json"),
    ("run3  hard-multi    (n=5)", f"{DOCS}/validation-study-multihard.json"),
    ("run4  feedback      (n=3)", f"{DOCS}/validation-study-feedback.json"),
    ("D5    bcb-hard code (n=2)", f"{DOCS}/validation-study-bcb.json"),
    ("run11 nano worker   (n=3)", f"{DOCS}/validation-study-multihard-nano.json"),
]


def _seed_caveat(path: str) -> str:
    """`` for a run whose seeds are independent draws, else a marker naming the measured
    overlap. NOT cosmetic: the interval and the sign-test p on that line both assume
    independent replications, and a shared task pool breaks that assumption in the
    direction that overstates significance."""
    meta = _meta(path)
    task_set = task_set_of(meta)
    if seeds_are_independent(task_set):
        return ""
    seeds = list(meta.get("seeds") or [])
    overlap = heldout_overlap(task_set, seeds, meta.get("n_heldout"))
    return f"  [!] seeds share {overlap:.0%} of held-out tasks — CI/p ANTI-CONSERVATIVE"


def _ceiling_hits(path: str) -> list[bool]:
    """Per seed: did BOTH arms reach the metric's maximum on held-out?

    A Δ of 0.0 there says the measurement ran out of headroom, not that the arms behave
    alike. Without this the p0 sweep's saturated level reads like its inert level.
    """
    try:
        runs = _runs(path)
    except (OSError, KeyError):
        return []
    return [
        r.get("B", {}).get("opt_heldout") == 1.0 and r.get("D", {}).get("opt_heldout") == 1.0
        for r in runs
    ]


def arms_optimized(path: str) -> dict[str, int]:
    """Per optimizing arm, the number of seeds on which it actually changed a gene.

    A Δ(D−B) is a D-vs-B contrast only if BOTH arms received their treatment. The budget
    guard in `run_validation_study` enforces that an arm CAN complete one accept cycle; it
    cannot enforce that a proposal is accepted, and the two are different. The split-size
    sweep's smallest row demonstrates the gap: at n_train=4 arm B accepted nothing on three of
    three seeds, so its Δ measures loop-level optimization against the *seed* genome.

    Arms A and C are excluded because neither is asked to change anything -- A evolves nothing
    and C's worker ignores the genome -- so counting them would report a permanent failure.
    """
    with open(path, encoding="utf-8") as fh:
        runs = json.load(fh)["runs"]
    out: dict[str, int] = {}
    for arm in ("B", "D"):
        if not all(isinstance(r.get(arm), dict) for r in runs):
            continue
        out[arm] = sum(1 for r in runs if r[arm].get("genes_changed"))
    return out


def is_valid_contrast(path: str) -> bool:
    """True when every optimizing arm accepted a change on at least one seed.

    Deliberately a low bar: it does not ask the arms to have optimized equally, only that
    neither is silently the seed genome. A False here means the run's Δ cannot be read as the
    contrast its label claims, whatever its interval says.
    """
    counts = arms_optimized(path)
    return bool(counts) and all(v > 0 for v in counts.values())


def _describe(label: str, d: list[float], path: str | None = None) -> str:
    """One line per run: point estimate, interval, sign test, and degenerate-seed count.

    `zeros` counts seeds whose Δ is exactly 0.0, and it is a load-bearing caveat with TWO
    distinct causes that must not be conflated. It was previously labelled as counting
    byte-identical genomes, which it never did: it counts equal SCORES.

      * No treatment: B and D optimized to byte-identical genomes, so the cassette served
        both arms the same completion and the 0.0 is an identity, not an observation. This is
        what the retracted runs suffered, and a bare interval of [0,0] hides it.
      * Ceiling: the arms differ but both hit the same capped score. The p0 sweep's highest
        level does exactly this -- on one seed arm B evolved `generate`, arm D evolved
        `generate` and `ensure_correct`, and BOTH scored 1.000, because the metric had no room
        left. Reading that as "the treatment did nothing" is precisely backwards.

    Distinguishing them needs the genomes, which only the newer runs release, so the count is
    reported as what it measures and the two causes are separated by `saturated` below.

    `path` adds the seed-independence marker. Optional only because two callers compare
    ACROSS runs; every single-run line passes it.
    """
    m, lo, hi = exact_resample_ci(d)
    zeros = sum(1 for x in d if x == 0.0)
    # A zero at the metric ceiling is a ceiling artifact, not an absent treatment.
    saturated = sum(1 for x in _ceiling_hits(path) if x) if path else 0
    seeds = [f"{x:+.3f}" for x in d]
    # An arm that accepted nothing is still the seed genome, so the line is not the contrast
    # its label claims. Marked here rather than left to the reader, because a Δ with an
    # interval and a p-value looks like a measurement whatever its arms did.
    one_sided = ""
    if path:
        counts = arms_optimized(path)
        starved = [a for a, v in counts.items() if v == 0]
        if starved:
            one_sided = f"  [!!] arm {'/'.join(starved)} accepted nothing — NOT a D-vs-B contrast"
    return (
        f"{label}: Δ={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]"
        f"  sign p={sign_test_p(d):.2f}  zeros={zeros}/{len(d)}"
        f"{f'  [ceiling on {saturated}]' if saturated else ''}"
        f"{_seed_caveat(path) if path else ''}{one_sided}"
    )


_CONTRAST_LEGEND = (
    "[!!] = an optimizing arm accepted no proposal on any seed, so it is still running the\n"
    "     SEED genome. The Δ on that line therefore compares the other arm against the seed,\n"
    "     not against its intended control, and must not be read as D-vs-B. The budget guard\n"
    "     enforces that an arm CAN fund an accept cycle; it cannot enforce that one is\n"
    "     accepted. This is the retraction's defect entering from the unwatched side."
)

_SEED_LEGEND = (
    "[!] = this run's loader shuffles a FIXED vendored pool, so a seed chooses a partition\n"
    "    rather than a fresh sample and the per-seed differences are positively correlated.\n"
    "    The exact-resample CI and the sign test both assume independent replications, so\n"
    "    on these lines both are anti-conservative (interval too narrow, p too small) and\n"
    "    must not be read as replication. multi/multi-hard regenerate their pool per seed\n"
    "    and are unaffected. A correct interval needs the PER-TASK scores, which runs now\n"
    "    release (ArmResult.opt_heldout_by_task) but none of these recorded ones carry."
)


def main() -> int:  # pragma: no cover - CLI convenience
    print("=== Δ(D−B) per-seed, mean, exact-resample 95% CI, sign test, zeros ===")
    print(
        "(zeros = seeds with Δ exactly 0.0. Two causes, do not conflate: identical\n"
        " genomes (no treatment), or both arms at the metric ceiling. [ceiling on N]\n"
        " marks the second.)"
    )
    for label, path in _NULL_RUNS:
        print(_describe(label, delta_arm(path), path))

    # The budget sweep is reported as its own block, ordered by effective search depth rather
    # than by budget, because depth is what it varies: above mmc=74 the proposal cap binds and
    # extra budget buys nothing.
    budget = [(lab, p) for lab, p in _BUDGET_RUNS if Path(p).exists()]
    if budget:
        print("\n=== budget sweep (D4): Δ(D−B) by effective search depth, n_train=8 fixed ===")
        print("depth = min(promotable cycles, proposal cap). The cap was 6 on every run before")
        print("this sweep and never bound; the deepest cell raises it to 12 and says so.")
        for label, path in budget:
            print(_describe(label, delta_arm(path), path))

        # A coarser readout than Δ, and worth printing beside it: if budget mattered anywhere, the
        # first place it should show is arms accepting more often as depth grows. NOT a
        # deterministic function of depth, though -- the observed pattern is non-monotone (a seed
        # can accept at a smaller depth and not at a larger one), because the search path depends
        # on sampled reflections. So read it as a second noisy signal that happens to be coarse,
        # not as a clean instrument.
        print("\nper-seed acceptance (did the arm change a gene?), by effective depth:")
        print(f"{'depth':>7}  {'arm B':<14} {'arm D':<14}")
        for label, path in budget:
            runs = _runs(path)
            cells = []
            for arm in ("B", "D"):
                marks = "".join(
                    "Y" if (r.get(arm, {}).get("genes_changed") or []) else "." for r in runs
                )
                cells.append(f"{marks:<14}")
            depth = label.split("depth")[-1].strip(" )")
            print(f"{depth:>7}  {cells[0]}{cells[1]}")
        print("(one character per seed, in order; Y = accepted at least one gene change)")
        if len(budget) < len(_BUDGET_RUNS):
            missing = [lab for lab, p in _BUDGET_RUNS if not Path(p).exists()]
            print(f"NOT YET BOUGHT: {', '.join(missing)}")
    print("\n=== effect: Δ(hi−lo) actionability, paired by seed (n=3) ===")
    print("NOTE: measured under --oracle-critique, which gates on the TRUE metric at")
    print("held-out scoring time too. This is a ceiling result: it says richer repair")
    print("feedback helps GIVEN a perfect checker, not that it helps in deployment.")
    d = delta_between(
        f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(d)
    excl = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
    seeds = [f"{x:+.3f}" for x in d]
    print(
        f"Δ(hi−lo)={m:+.3f}  seeds={seeds}  95%CI[{lo:+.3f},{hi:+.3f}]  ({excl})"
        f"  sign p={sign_test_p(d):.2f}"
        f"{_seed_caveat(f'{DOCS}/validation-study-fbacthi.json')}"
    )

    dc = delta_between(
        f"{DOCS}/validation-study-bcb-acthi.json", f"{DOCS}/validation-study-bcb-actlo.json"
    )
    print(f"\n=== code actionability: Δ(hi−lo) arm-A held-out, BigCodeBench-Hard (n={len(dc)}) ===")
    mc, loc, hic = exact_resample_ci(dc)
    exclc = "excludes 0" if loc > 0 or hic < 0 else "includes 0 (suggestive, not significant)"
    seedsc = [f"{x:+.3f}" for x in dc]
    print(
        f"Δ(hi−lo)={mc:+.3f}  seeds={seedsc}  95%CI[{loc:+.3f},{hic:+.3f}]  ({exclc})"
        f"{_seed_caveat(f'{DOCS}/validation-study-bcb-acthi.json')}"
    )

    print("\n=== actuator experiment: Δ(D−B) aiming GEPA at the fixer's feedback gene ===")
    for label, path in (
        ("actuator small (n=2, 8/10, mmc30)", f"{DOCS}/validation-study-bcb-actuator.json"),
        ("actuator powered (n=3, 20/20, mmc50)", f"{DOCS}/validation-study-bcb-actuator-big.json"),
    ):
        da = delta_arm(path)
        _, loa, hia = exact_resample_ci(da)
        # A negative effect "excludes zero" only when its upper bound < 0. With the
        # percentile index corrected, the powered run's upper bound is exactly 0.0 (the
        # all-zero resample is the largest atom and carries 1/27 > alpha/2), so this run
        # does NOT exclude zero — the earlier claim that it did was an artifact of
        # dropping that atom. A degenerate [0,0] interval is reported as "point zero".
        excla = "excludes zero" if hia < 0 else ("point zero" if loa == hia == 0 else "includes 0")
        print(f"{_describe(label, da, path)}  ({excla})")
    print(f"\n{_CONTRAST_LEGEND}")
    print(f"\n{_SEED_LEGEND}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
