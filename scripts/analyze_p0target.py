#!/usr/bin/env python
"""The pre-registered analysis of the target-p0 study. Zero cost, offline.

Reads the three four-arm runs (levels low, mid, high of `docs/p0-manifests/study.json`) and
their cassettes, and reports exactly the quantities the 2026-10-25 pre-registration defines:

- Manipulation checks. M1, realized p0: the share of arm A's first drafts on held-out that
  already pass, from the runs' own draws (not the calibration). M2, matched baseline: arm A's
  held-out means. M3, treatment applied: seeds on which arm D changed `ensure_correct`.
- H1 (primary): the swing S = mean Δ_mid − mean Δ_low, Δ = D.opt_heldout − B.opt_heldout per
  seed, with a percentile bootstrap that resamples seeds independently within each level
  (10 000 resamples, seed 0) and Welch's t. SUPPORTED / CONTRADICTED / INCONCLUSIVE by the
  registered rule.
- H2: the ordering of the three level means, and S_high = mean Δ_high − mean Δ_low.
- H3: the evolved critique's PASS rate on held-out first critiques per level, and its pooled
  false PASS count.
- H4: the decomposition Δ = draft term + loop term, where the draft term compares the two
  winners' first drafts and the loop term what each loop added after them.

Every replayed number is recomputed from the cassette with the winners' own gene text, so a
replay miss raises instead of shrinking a mean.

    python scripts/analyze_p0target.py
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from collections.abc import Sequence
from pathlib import Path
from typing import Any

N_BOOT = 10_000
LEVELS = ("low", "mid", "high")
MANIFEST = "docs/p0-manifests/study.json"
RESULTS = "docs/p0target/validation-study-p0target-{level}.json"
CASSETTE = "tests/fixtures/cassettes/study-p0target-{level}.json"
N_HELDOUT = 10
VERIFY_GENE = "ensure_correct"


def percentile_indices(n_boot: int) -> tuple[int, int]:
    """Indices of the 2.5th and 97.5th percentiles in a sorted list of n_boot resamples."""
    return int(math.floor(0.025 * (n_boot - 1))), int(math.ceil(0.975 * (n_boot - 1)))


def chi2_presence_by_level(present: Sequence[int], totals: Sequence[int]) -> tuple[float, float]:
    """Pearson chi-square for a constraint's presence across three levels (a 2x3 table), df = 2.

    For df = 2 the survival function is exactly exp(-x/2), so no scipy is needed."""
    if len(present) != 3 or len(totals) != 3:
        raise ValueError("expects three levels")
    absent = [t - k for k, t in zip(present, totals, strict=True)]
    n = sum(totals)
    stat = 0.0
    for row in (present, absent):
        r = sum(row)
        for o, t in zip(row, totals, strict=True):
            e = r * t / n
            stat += 0.0 if e == 0 else (o - e) ** 2 / e
    return stat, math.exp(-stat / 2)


def beta_binomial_mom(counts: Sequence[int], k: int) -> tuple[float, float]:
    """Method-of-moments Beta(a, b) prior from per-task success counts out of k draws."""
    ps = [c / k for c in counts]
    m = st.mean(ps)
    v = st.pvariance(ps)
    vp = (v - (m - m * m) / k) / (1 - 1 / k)
    if vp <= 0:
        raise ValueError("no between-task variance left after removing binomial noise")
    ab = m * (1 - m) / vp - 1
    return m * ab, (1 - m) * ab


def swing(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    """S = mean(a) − mean(b); percentile bootstrap resampling each group independently."""
    rng = random.Random(seed)
    s = st.mean(a) - st.mean(b)
    boots = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        boots.append(st.mean(ra) - st.mean(rb))
    boots.sort()
    i, j = percentile_indices(n_boot)
    return {"S": s, "ci": (boots[i], boots[j]), "n_boot": n_boot}


def pooled(x: Sequence[float], *, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Mean of independent seed deltas with a percentile bootstrap over seeds. EXPLORATORY in
    this study: it pools levels whose p0 did not separate, which the registration did not plan."""
    rng = random.Random(seed)
    boots = sorted(st.mean(x[rng.randrange(len(x))] for _ in x) for _ in range(n_boot))
    i, j = percentile_indices(n_boot)
    return {"mean": st.mean(x), "ci": (boots[i], boots[j]), "n": len(x), "sd": st.stdev(x)}


def welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t and Welch-Satterthwaite df for mean(a) − mean(b)."""
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return t, df


def welch_p(t: float, df: float) -> float:
    """Two-sided p from Student's t by numerical integration (no scipy in this venv)."""

    def pdf(x: float) -> float:
        c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        return c * (1 + x * x / df) ** (-(df + 1) / 2)

    x, n = abs(t), 20_000
    h = x / n
    area = sum(pdf((i + 0.5) * h) for i in range(n)) * h  # integral 0..|t|
    return max(0.0, 1.0 - 2.0 * area)


def verdict(s: float, ci: tuple[float, float]) -> str:
    if ci[0] > 0 and s > 0:
        return "SUPPORTED"
    if ci[1] < 0 and s < 0:
        return "CONTRADICTED"
    return "INCONCLUSIVE"


def decompose(
    *, draft_b: float, final_b: float, draft_d: float, final_d: float
) -> dict[str, float]:
    draft = draft_d - draft_b
    loop = (final_d - draft_d) - (final_b - draft_b)
    return {"delta": final_d - final_b, "draft": draft, "loop": loop}


def _replay_level(level: str) -> dict[str, Any]:  # pragma: no cover - needs the released run
    from whystuck.eval.cassette import Cassette
    from whystuck.eval.llm_worker import make_self_refine_run_task
    from whystuck.eval.multi_constraint import mc_feedback, mc_pass, mc_score
    from whystuck.eval.study import seed_genome
    from whystuck.eval.task_manifest import load_manifest, manifest_split

    res = json.loads(Path(RESULTS.format(level=level)).read_text())
    manifest = load_manifest(MANIFEST)
    seeds = res["meta"]["seeds"]
    out: dict[str, Any] = {"seeds": seeds, "per_seed": []}
    with Cassette(CASSETTE.format(level=level), mode="replay") as cas:

        def _unreachable(_p: str) -> str:
            raise AssertionError("unreachable in replay mode")

        run_task = make_self_refine_run_task(
            cas.wrap(_unreachable, name="worker"), mc_score, feedback=mc_feedback, max_retries=1
        )
        for seed, run in zip(seeds, res["runs"], strict=True):
            heldout = manifest_split(manifest, level, "heldout", seed=seed, n=N_HELDOUT)
            row: dict[str, Any] = {"seed": seed}
            for arm in ("A", "B", "D"):
                genes = run[arm].get("final_genes") or {}
                genome = seed_genome(f"study-{arm}", evolvable=set(), overrides=genes or None)
                drafts, finals, passes, false_pass = [], [], 0, 0
                for task in heldout:
                    tr = run_task(genome, task)
                    first = next(s for s in tr.steps if s.step_id == "generate-0").output_summary
                    crit = next(s for s in tr.steps if s.step_id == "critique-0").output_summary
                    drafts.append((mc_score(first or "", task), mc_pass(first or "", task)))
                    finals.append(float(tr.metadata["eval_score"]))
                    said_pass = (crit or "").strip().upper().startswith("PASS")
                    passes += said_pass
                    false_pass += said_pass and not mc_pass(first or "", task)
                row[arm] = {
                    "draft": st.mean(d for d, _ in drafts),
                    "p0": st.mean(1.0 if ok else 0.0 for _, ok in drafts),
                    "final": st.mean(finals),
                    "released_final": run[arm]["opt_heldout"],
                    "pass_rate": passes / len(heldout),
                    "false_pass": false_pass,
                    "genes_changed": run[arm].get("genes_changed") or [],
                    "first_pass_by_task": {
                        t.id: ok for t, (_, ok) in zip(heldout, drafts, strict=True)
                    },
                }
                if abs(row[arm]["final"] - run[arm]["opt_heldout"]) > 1e-9:
                    raise AssertionError(
                        f"{level} seed {seed} arm {arm}: replayed held-out {row[arm]['final']} "
                        f"!= released {run[arm]['opt_heldout']}; the replay is not the run"
                    )
            out["per_seed"].append(row)
    return out


def _diagnose(levels: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - needs the run
    """Why the levels did not separate, and what else differs between them.

    Composition: the registered level-by-level constraint mix, per generator tier, with a
    per-constraint chi-square of presence against level (df = 2). Shrinkage: a method-of-moments
    beta-binomial prior per tier from the K = 3 calibration counts, the level means it predicts,
    and the realized first-draft pass rate of each calibration count on the runs' fresh draws."""
    from whystuck.eval.task_manifest import load_manifest

    man = load_manifest(MANIFEST)
    cal: dict[str, Any] = {}
    for name in ("hard", "L2"):
        cal.update(json.loads(Path(f"docs/p0-manifests/full-1001-{name}-stats.json").read_text()))

    def tier(tid: str) -> str:
        return "l2" if "x-exact_word_count" in tid else "hard"

    out: dict[str, Any] = {"composition": {}, "shrinkage": {}}
    for tr in ("hard", "l2"):
        totals, present = [], {}
        for lv in LEVELS:
            ids = [
                i
                for sd in man.levels[lv].values()
                for sp in ("train", "heldout")
                for i in sd[sp]
                if tier(i) == tr
            ]
            totals.append(len(ids))
            for i in ids:
                for cid in man.tasks[i].payload["instruction_id_list"]:
                    present.setdefault(cid, [0, 0, 0])[LEVELS.index(lv)] += 1
        rows = {}
        for cid, k in sorted(present.items()):
            stat, pval = chi2_presence_by_level(k, totals)
            rows[cid] = {
                "share": [k[i] / totals[i] for i in range(3)],
                "chi2": stat,
                "p": pval,
            }
        out["composition"][tr] = {"tasks_per_level": totals, "constraints": rows}

        counts = [round(v["p_first"] * 3) for t, v in cal.items() if tier(t) == tr]
        a, b = beta_binomial_mom(counts, 3)
        post = {
            t: (a + round(v["p_first"] * 3)) / (a + b + 3) for t, v in cal.items() if tier(t) == tr
        }
        realized: dict[str, list[float]] = {}
        by_count: dict[int, list[float]] = {}
        for lv in LEVELS:
            for r in levels[lv]["per_seed"]:
                for tid, ok in r["A"]["first_pass_by_task"].items():
                    if tier(tid) != tr:
                        continue
                    realized.setdefault(lv, []).append(1.0 if ok else 0.0)
                    by_count.setdefault(round(cal[tid]["p_first"] * 3), []).append(
                        1.0 if ok else 0.0
                    )
        heldout_ids = {
            lv: [i for sd in man.levels[lv].values() for i in sd["heldout"] if tier(i) == tr]
            for lv in LEVELS
        }
        out["shrinkage"][tr] = {
            "beta_a": a,
            "beta_b": b,
            "levels": {
                lv: {
                    "calibrated": st.mean(cal[i]["p_first"] for i in heldout_ids[lv]),
                    "posterior": st.mean(post[i] for i in heldout_ids[lv]),
                    "realized": st.mean(realized[lv]),
                }
                for lv in LEVELS
            },
            "realized_by_calibration_count": {
                str(c): {"n": len(v), "pass": st.mean(v)} for c, v in sorted(by_count.items())
            },
        }
    return out


def main() -> int:  # pragma: no cover - CLI convenience
    levels = {lv: _replay_level(lv) for lv in LEVELS if Path(RESULTS.format(level=lv)).exists()}
    deltas = {
        lv: [r["D"]["final"] - r["B"]["final"] for r in v["per_seed"]] for lv, v in levels.items()
    }
    report: dict[str, Any] = {"levels": {}}
    for lv, v in levels.items():
        rows = v["per_seed"]
        dec = [
            decompose(
                draft_b=r["B"]["draft"], final_b=r["B"]["final"],
                draft_d=r["D"]["draft"], final_d=r["D"]["final"],
            )
            for r in rows
        ]  # fmt: skip
        report["levels"][lv] = {
            "n_seeds": len(rows),
            "M1_realized_p0": st.mean(r["A"]["p0"] for r in rows),
            "M2_armA_heldout": st.mean(r["A"]["final"] for r in rows),
            "M3_seeds_D_changed_verify": sum(VERIFY_GENE in r["D"]["genes_changed"] for r in rows),
            "B_mean": st.mean(r["B"]["final"] for r in rows),
            "D_mean": st.mean(r["D"]["final"] for r in rows),
            "delta_mean": st.mean(deltas[lv]),
            "delta_per_seed": deltas[lv],
            "delta_sd": st.stdev(deltas[lv]),
            "H3_seeds_D_pass_gt0": sum(r["D"]["pass_rate"] > 0 for r in rows),
            "H3_D_pass_rate": st.mean(r["D"]["pass_rate"] for r in rows),
            "H3_D_false_pass_pooled": sum(r["D"]["false_pass"] for r in rows),
            "H3_A_pass_rate": st.mean(r["A"]["pass_rate"] for r in rows),
            "H4_draft_term": st.mean(d["draft"] for d in dec),
            "H4_loop_term": st.mean(d["loop"] for d in dec),
            "H3_A_false_pass_pooled": sum(r["A"]["false_pass"] for r in rows),
        }
    if levels:
        report["diagnosis"] = _diagnose(levels)
    if "low" in deltas and "mid" in deltas:
        s = swing(deltas["mid"], deltas["low"])
        t, df = welch(deltas["mid"], deltas["low"])
        report["H1"] = {**s, "welch_t": t, "welch_df": df, "welch_p": welch_p(t, df),
                        "verdict": verdict(s["S"], s["ci"])}  # fmt: skip
    if "low" in deltas and "high" in deltas:
        report["H2_S_high"] = swing(deltas["high"], deltas["low"])
        means = [st.mean(deltas[lv]) for lv in LEVELS if lv in deltas]
        report["H2_monotone"] = all(x <= y for x, y in zip(means, means[1:], strict=False))
    if all(lv in deltas for lv in LEVELS):
        allx = [x for lv in LEVELS for x in deltas[lv]]
        report["exploratory_pooled"] = {
            **pooled(allx),
            "mean_realized_p0": st.mean(report["levels"][lv]["M1_realized_p0"] for lv in LEVELS),
        }
    out = Path("docs/p0target/analysis.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
