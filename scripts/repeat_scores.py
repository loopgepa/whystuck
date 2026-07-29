#!/usr/bin/env python
"""Re-score released genomes K times to put a sampling interval on Δ (D7 / P1).

WHY THIS IS THE HIGHEST-VALUE PAID EXPERIMENT LEFT. No interval in this study contains
within-condition sampling variance: the worker drew one sample per task at the provider's
default temperature, so every reported Δ is a single draw. An offline estimate from runs that
happened to re-score the same seed genome puts that noise at σ(Δ) ≈ 0.03 over a 30-task mean
(`analyze_measurement_noise.py`) --- which is 1.4σ to 2.1σ of every headline the paper reports.
K replicates shrink it by √K, so a few dollars decides results that are currently marginal.

It needs no re-optimization. GEPA's output is already released as `final_genes`, so a replicate
is just the existing scoring path run again on a released genome: `--arms A --seed-gene`, which
is exactly the mechanism the C1 verifier ablation used. Two properties make it correct:

  * **The salt.** Repeated scorings of one genome are IDENTICAL calls, so record mode would
    dedupe them and hand K replicates one cached draw --- reporting zero sampling variance,
    the most flattering artifact available to a study about flattering artifacts. `--sample k`
    enters the cassette key without reaching the provider.
  * **No pinned temperature, deliberately.** The point is to estimate the noise in the numbers
    the paper ALREADY reports, and those were drawn at the provider's default. Pinning a value
    would measure the noise of a distribution the study never sampled from.

Resumable by construction: a job whose output JSON already exists is skipped, so an
interruption never re-buys a completed replicate.

    python scripts/repeat_scores.py                 # dry run: plan + cost, buys nothing
    python scripts/repeat_scores.py --k 5 --yes     # execute
    python scripts/repeat_scores.py --aggregate     # read results back, no API needed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

DOCS = "docs"
OUT_DIR = f"{DOCS}/repeat"
CASSETTE_DIR = "tests/fixtures/cassettes/repeat"
EVAL_PY = "scratchpad/evalvenv/bin/python"
CLI = "scripts/run_validation_study.py"

#: (level, released run slug, exclusions) — the p0 sweep, where the sign flip lives.
LEVELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("L1", "c2b", ()),
    ("L2", "p0-L2", ("exact_word_count",)),
    ("L3", "p0-L3", ("exact_word_count", "word_max_length")),
)

#: Arms to re-score. SEED is the unoptimized genome, kept because it is the only job whose
#: answer we already know from the cross-run estimate: if the K-sample spread does not
#: reproduce σ ≈ 0.03, the salt is not working and every other number here is void.
ARMS = ("B", "D", "SEED")
N_HELDOUT = 10
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Job:
    level: str
    slug: str
    exclusions: tuple[str, ...]
    arm: str
    seed: int
    k: int

    @property
    def tag(self) -> str:
        return f"{self.level}-{self.arm}-s{self.seed}-k{self.k}"

    @property
    def out(self) -> str:
        return f"{OUT_DIR}/{self.tag}.json"

    @property
    def cassette(self) -> str:
        return f"{CASSETTE_DIR}/{self.tag}.json"

    @property
    def done(self) -> bool:
        return Path(self.out).exists()


def released_genes(slug: str, arm: str, seed: int) -> dict[str, str]:
    """The genome to re-score. `SEED` means the unoptimized one, i.e. no overrides."""
    if arm == "SEED":
        return {}
    runs = json.loads(Path(f"{DOCS}/validation-study-{slug}.json").read_text())["runs"]
    return {k: str(v) for k, v in runs[seed][arm]["final_genes"].items()}


def plan(k: int) -> list[Job]:
    return [
        Job(level, slug, excl, arm, seed, rep)
        for level, slug, excl in LEVELS
        for arm in ARMS
        for seed in SEEDS
        for rep in range(k)
    ]


def parse_shard(spec: str) -> tuple[int, int]:
    """`"i/n"` -> (i, n). Raises on anything else rather than guessing a default.

    A misread shard spec is the one failure mode that costs money: two workers that both
    believe they are shard 0 buy every job twice.
    """
    parts = spec.split("/")
    if len(parts) != 2 or not all(p.strip().lstrip("-").isdigit() for p in parts):
        raise ValueError(f"--shard wants 'i/n', got {spec!r}")
    return int(parts[0]), int(parts[1])


def shard(jobs: list[Job], index: int, count: int) -> list[Job]:
    """The `index`-th of `count` disjoint slices of `jobs`, by position.

    Positional striding rather than hashing: it is inspectable, it needs no shared state, and
    the union is provably the whole plan (see the tests). Every job owns its output file and its
    own cassette, so parallel workers never contend -- the only shared resource is the API key.
    """
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"shard {index}/{count} is not a valid slice")
    return [j for k, j in enumerate(jobs) if k % count == index]


def argv_for(job: Job, gene_files: dict[str, str], *, execute: bool) -> list[str]:
    """The exact CLI invocation for one replicate.

    `--arms A` scores a fixed genome and runs no optimizer, so the budget guard is inert and
    `--max-metric-calls` only has to exceed the held-out count.
    """
    argv = [
        EVAL_PY,
        CLI,
        "--task-set",
        "multi-hard",
        "--arms",
        "A",
        "--seeds",
        str(job.seed),
        "--n-train",
        "8",
        "--n-heldout",
        str(N_HELDOUT),
        "--max-metric-calls",
        "28",
        "--loop-max-retries",
        "1",
        "--module-selector",
        "all",
        "--reflection-model",
        "openai/gpt-5.6-luna",
        "--sample",
        str(job.k),
        "--record",
        job.cassette,
        "--out",
        job.out,
    ]
    for constraint in job.exclusions:
        argv += ["--exclude-constraint", constraint]
    for gene, path in sorted(gene_files.items()):
        argv += ["--seed-gene", f"{gene}={path}"]
    if execute:
        argv.append("--yes")
    return argv


def verify_argv_for(job: Job, gene_files: dict[str, str], *, out: str) -> list[str]:
    """The same invocation in replay mode, writing somewhere other than the committed output.

    Registering 135 replicates one-by-one in `verify_reproducibility.py` does not scale, and a
    registry nobody maintains stops being a check. The driver already owns the exact argv, so it
    re-derives it here instead. Two properties make the result meaningful rather than decorative:
    `--yes` is absent, so a verification pass cannot spend even if the cassette is incomplete;
    and `--out` is redirected, so the comparison cannot be satisfied by overwriting the artifact
    under test with a fresh copy of itself.
    """
    argv = [a for a in argv_for(job, gene_files, execute=False)]
    argv[argv.index("--out") + 1] = out
    return argv + ["--replay"]


# ------------------------------------------------------------------------------ aggregation


#: Two-sided Student-t critical values, by degrees of freedom. Tabulated rather than computed
#: so the script needs no special-function implementation, and pinned by a test so a typo cannot
#: quietly turn a null into a result. K replicates give df = K - 1.
T_CRIT_05 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T_CRIT_10 = {2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


@dataclass(frozen=True)
class LevelResult:
    level: str
    k: int
    replicate_deltas: list[float]
    n_tasks: int

    @property
    def delta(self) -> float:
        return mean(self.replicate_deltas)

    @property
    def sd_between_replicates(self) -> float:
        """The sampling noise on a single-draw Δ — what the paper's headlines omit."""
        return pstdev(self.replicate_deltas)

    @property
    def standard_error(self) -> float:
        """sd/√K: the noise on the K-replicate mean, i.e. what K buys."""
        return self.sd_between_replicates / len(self.replicate_deltas) ** 0.5

    @property
    def df(self) -> int:
        return len(self.replicate_deltas) - 1

    def t_stat(self) -> float:
        """The mean over its own standard error. At K=5 this is Student-t with 4 df."""
        se = self.standard_error
        return abs(self.delta) / se if se else float("inf")

    def verdict(self) -> str:
        """Significance against the t critical values, not the normal ones.

        At K=5 the normal 1.96 would call a 2.2-se effect significant while t(4) does not. Two
        of the three effects here land in exactly that gap, so the distinction is the result.
        """
        t, df = self.t_stat(), self.df
        if df not in T_CRIT_05:
            return "df out of table"
        if t > T_CRIT_05[df]:
            return "p < 0.05"
        if t > T_CRIT_10[df]:
            return "p < 0.10"
        return "not significant"

    def in_sigma(self) -> float:
        """Retained name for the t statistic, which is what earlier drafts called sigma."""
        return self.t_stat()


def _vector(path: str) -> dict[str, float]:
    run = json.loads(Path(path).read_text())["runs"][0]
    return run["A"]["seed_heldout_by_task"]


def aggregate(k: int) -> list[LevelResult]:
    """Per level, the Δ(D−B) of each replicate, pooled over seeds.

    A replicate's Δ is computed within that replicate: arm D's scores minus arm B's on the
    same tasks and the same salt. Pooling across replicates first would average the noise away
    before measuring it.
    """
    out: list[LevelResult] = []
    for level, slug, excl in LEVELS:
        deltas: list[float] = []
        n_tasks = 0
        for rep in range(k):
            b_scores: list[float] = []
            d_scores: list[float] = []
            for seed in SEEDS:
                jb = Job(level, slug, excl, "B", seed, rep)
                jd = Job(level, slug, excl, "D", seed, rep)
                if not (jb.done and jd.done):
                    b_scores = d_scores = []
                    break
                vb, vd = _vector(jb.out), _vector(jd.out)
                if sorted(vb) != sorted(vd):
                    raise ValueError(
                        f"{jb.tag} and {jd.tag} scored different held-out task ids; a Δ across "
                        "different task sets is not a Δ"
                    )
                b_scores += [vb[t] for t in sorted(vb)]
                d_scores += [vd[t] for t in sorted(vd)]
            if b_scores and d_scores:
                deltas.append(mean(d_scores) - mean(b_scores))
                n_tasks = len(b_scores)
        if deltas:
            out.append(LevelResult(level, len(deltas), deltas, n_tasks))
    return out


def sigma_by_genome(k: int) -> dict[tuple[str, str], float]:
    """(level, arm) -> sd of that genome's 30-task mean across the K replicates.

    THE EXPERIMENT'S OWN FINDING. The offline estimate assumed one sigma for the whole study,
    which is what every interval in the paper was then scaled by. Measured directly, sigma is a
    property of the GENOME: a critique that refines every draft issues more sampled calls than
    one that accepts some, and an arm near the metric ceiling has less room to vary at all. A
    single study-wide sigma is therefore an approximation, and this returns the spread it hides.
    """
    out: dict[tuple[str, str], float] = {}
    for level, slug, excl in LEVELS:
        for arm in ARMS:
            means: list[float] = []
            for rep in range(k):
                vals: list[float] = []
                for seed in SEEDS:
                    job = Job(level, slug, excl, arm, seed, rep)
                    if not job.done:
                        vals = []
                        break
                    v = _vector(job.out)
                    vals += [v[t] for t in sorted(v)]
                if vals:
                    means.append(mean(vals))
            if len(means) > 1:
                out[(level, arm)] = pstdev(means)
    return out


def seed_replicate_spread(k: int) -> tuple[int, float]:
    """(replicates found, sd of the SEED genome's 30-task mean across replicates).

    The control. The cross-run estimate says a single-draw 30-task mean of this genome moves
    with σ ≈ 0.03. If K salted replicates reproduce that, the salt works and the loop is
    sampling. If it comes back ≈ 0, record mode is serving one cached draw and every other
    number in this file is void.
    """
    means: list[float] = []
    for rep in range(k):
        vals: list[float] = []
        for level, slug, excl in LEVELS[:1]:  # L1 only: the unexcluded regime
            for seed in SEEDS:
                job = Job(level, slug, excl, "SEED", seed, rep)
                if not job.done:
                    vals = []
                    break
                v = _vector(job.out)
                vals += [v[t] for t in sorted(v)]
        if vals:
            means.append(mean(vals))
    return len(means), (pstdev(means) if len(means) > 1 else float("nan"))


def _run_jobs(jobs: list[Job], *, execute: bool) -> int:
    import tempfile

    todo = [j for j in jobs if not j.done]
    print(f"{len(jobs)} job(s); {len(jobs) - len(todo)} already done, {len(todo)} to run.")
    if not execute:
        print("\nDRY RUN — nothing is bought. One representative invocation:\n")
        if todo:
            sample = todo[0]
            print("  " + " ".join(argv_for(sample, {"generate": "<tmp>"}, execute=False)))
        print(
            f"\nEach job scores {N_HELDOUT} held-out tasks through the loop "
            f"({N_HELDOUT} x (1 + 2R) calls at R=1). The committed p0 calibration runs are the "
            f"price anchor: 3 seeds x {N_HELDOUT} tasks cost $0.09, so one job is ~$0.03 and "
            f"{len(todo)} jobs are ~${0.03 * len(todo):.2f}."
        )
        print("Re-run with --yes to execute. Completed jobs are skipped, so it is resumable.")
        return 0

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CASSETTE_DIR).mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(todo, 1):
        # Re-check at the moment of spending, not only when `todo` was built. With parallel
        # shards the list can be minutes stale, and the cost of a stale entry is a bought
        # replicate, not a wasted loop iteration. Belt and braces on top of the partition.
        if job.done:
            print(f"[{i}/{len(todo)}] {job.tag} already done since planning, skipping")
            continue
        genes = released_genes(job.slug, job.arm, job.seed)
        with tempfile.TemporaryDirectory() as d:
            files = {}
            for gene, text in genes.items():
                path = Path(d) / f"{gene}.txt"
                path.write_text(text)
                files[gene] = str(path)
            argv = argv_for(job, files, execute=True)
            print(f"[{i}/{len(todo)}] {job.tag}", flush=True)
            proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            print(f"FAILED on {job.tag}; stopping so the spend is not compounded.", file=sys.stderr)
            return 1
    return 0


def verify_all(k: int) -> tuple[int, list[str]]:
    """Replay every completed replicate from its own cassette. Returns (checked, failures).

    Offline and free. What it proves is narrow but is exactly the property the paper needs: the
    numbers in `docs/repeat/` are the numbers those cassettes produce under HEAD, so a later
    change to the scoring path cannot silently move a published replicate.
    """
    import tempfile

    done = [j for j in plan(k) if j.done]
    failures: list[str] = []
    for i, job in enumerate(done, 1):
        genes = released_genes(job.slug, job.arm, job.seed)
        with tempfile.TemporaryDirectory() as d:
            files = {}
            for gene, text in genes.items():
                path = Path(d) / f"{gene}.txt"
                path.write_text(text)
                files[gene] = str(path)
            out = str(Path(d) / "replay.json")
            argv = verify_argv_for(job, files, out=out)
            proc = subprocess.run(argv, capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f"{job.tag}: replay exited {proc.returncode}")
                print(f"[{i}/{len(done)}] {job.tag} REPLAY FAILED", flush=True)
                continue
            want = json.loads(Path(job.out).read_text())["runs"][0]["A"]["seed_heldout_by_task"]
            got = json.loads(Path(out).read_text())["runs"][0]["A"]["seed_heldout_by_task"]
        ok = want == got
        if not ok:
            failures.append(f"{job.tag}: replay produced different per-task scores")
        print(f"[{i}/{len(done)}] {job.tag} {'ok' if ok else 'MISMATCH'}", flush=True)
    return len(done), failures


def _report(k: int) -> int:
    results = aggregate(k)
    n_seed, sd_seed = seed_replicate_spread(k)
    print("=== control: does the salt actually produce independent draws? ===")
    print(f"SEED genome, {n_seed} replicate(s), sd of its 30-task mean = {sd_seed:.4f}")
    print("Cross-run estimate for the SAME quantity (one draw of a 30-task mean) is 0.0197,")
    print("i.e. 0.153/sqrt(30) with drift between recording dates included. A value near 0 here")
    print("would mean record mode served one cached draw to every replicate.")
    print("The numbers below are void if that happens.\n")
    if not results:
        print("No completed B/D replicate pairs yet.")
        return 0
    print("=== Δ(D−B) with a repeated-sample interval ===")
    print(
        f"{'level':6} {'K':>3} {'Δ':>9} {'sd(draw)':>10} {'se(mean)':>10} "
        f"{'t':>6} {'df':>3} verdict"
    )
    for r in results:
        print(
            f"{r.level:6} {r.k:>3} {r.delta:>+9.4f} {r.sd_between_replicates:>10.4f} "
            f"{r.standard_error:>10.4f} {r.t_stat():>6.2f} {r.df:>3} {r.verdict()}"
        )
    print(f"\n(t critical values at df=4: {T_CRIT_10[4]} for p<0.10, {T_CRIT_05[4]} for p<0.05.")
    print("A normal approximation would use 1.96 and call two of these significant.)")

    sig = sigma_by_genome(len(results[0].replicate_deltas))
    if sig:
        print("\n=== sigma is a property of the GENOME, not of the study ===")
        print("sd of one 30-task-mean draw, per (level, arm):")
        print(f"{'level':6} " + " ".join(f"{a:>8}" for a in ARMS))
        for level, _, _ in LEVELS:
            row = " ".join(f"{sig.get((level, a), float('nan')):>8.4f}" for a in ARMS)
            print(f"{level:6} {row}")
        lo, hi = min(sig.values()), max(sig.values())
        print(f"\nrange {lo:.4f} to {hi:.4f} ({hi / lo:.1f}x). Any single study-wide sigma is an")
        print("approximation; the offline cross-run estimate assumed one.")
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5, help="replicates per genome (default 5)")
    p.add_argument("--yes", action="store_true", help="execute; without it this is a dry run")
    p.add_argument("--aggregate", action="store_true", help="read results back, no API calls")
    p.add_argument(
        "--verify",
        action="store_true",
        help="replay every completed replicate from its cassette, offline, and compare ($0)",
    )
    p.add_argument(
        "--shard",
        default=None,
        help="'i/n': run only the i-th of n disjoint slices, so n workers can run at once",
    )
    args = p.parse_args(argv)
    if args.verify:
        checked, failures = verify_all(args.k)
        if failures:
            print(f"\n{len(failures)} of {checked} replicate(s) did NOT reproduce:")
            for f in failures:
                print(f"  {f}")
            return 1
        print(f"\nALL {checked} COMPLETED REPLICATES REPRODUCE FROM CASSETTE ($0, offline).")
        return 0
    if args.aggregate:
        return _report(args.k)
    jobs = plan(args.k)
    if args.shard:
        index, count = parse_shard(args.shard)
        jobs = shard(jobs, index, count)
        print(f"shard {index}/{count}: {len(jobs)} of {len(plan(args.k))} job(s)")
    rc = _run_jobs(jobs, execute=args.yes)
    return _report(args.k) if (rc == 0 and args.yes) else rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
