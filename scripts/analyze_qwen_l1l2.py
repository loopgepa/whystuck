#!/usr/bin/env python
"""The pre-registered analysis of the third-worker L1/L2 test (2026-10-26). Zero cost, offline.

Reads `docs/qwen-l1l2/validation-study-qwen-{L1,L2}.json` and their cassettes, and reports what
`docs/superpowers/specs/2026-10-26-qwen-l1-l2-replication-preregistration.md` defines: the swing
S = mean Δ_L2 − mean Δ_L1 with the seed bootstrap and Welch's t (the same tested functions the
target-p0 analysis uses), each level's Δ with its own interval, and the manipulation checks M1
(realized p0 from arm A's held-out first drafts, by strict replay), M2 (arm A held-out mean and the
0.95 saturation flag) and M3 (seeds on which arm D changed `ensure_correct`), plus the evolved
critique's PASS rate and false PASSes. A replayed held-out score that differs from the released
one raises.

    python scripts/analyze_qwen_l1l2.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_p0target import pooled, swing, verdict, welch, welch_p  # noqa: E402

LEVELS = {"L1": frozenset(), "L2": frozenset({"exact_word_count"})}
RESULTS = "docs/qwen-l1l2/validation-study-qwen-{level}.json"
CASSETTE = "tests/fixtures/cassettes/study-qwen-l1l2-{level}.json"
VERIFY_GENE = "ensure_correct"
SATURATION = 0.95


def _replay(level: str) -> dict[str, Any]:  # pragma: no cover - needs the released run
    from whystuck.eval.cassette import Cassette
    from whystuck.eval.llm_worker import make_self_refine_run_task
    from whystuck.eval.multi_constraint import (
        load_multi_constraint,
        mc_feedback,
        mc_pass,
        mc_score,
    )
    from whystuck.eval.study import seed_genome

    res = json.loads(Path(RESULTS.format(level=level)).read_text())
    meta = res["meta"]
    n_heldout = meta["n_heldout"]
    pool = max(64, 2 * max(meta["n_train"], n_heldout) + 2)  # the runner's own rule
    rows = []
    with Cassette(CASSETTE.format(level=level), mode="replay") as cas:

        def _unreachable(_p: str) -> str:
            raise AssertionError("unreachable in replay mode")

        run_task = make_self_refine_run_task(
            cas.wrap(_unreachable, name="worker"), mc_score, feedback=mc_feedback, max_retries=1
        )
        for seed, run in zip(meta["seeds"], res["runs"], strict=True):
            heldout = load_multi_constraint(
                "heldout", seed=seed, n=n_heldout, pool=pool, hard=True, exclude=LEVELS[level]
            )
            row: dict[str, Any] = {"seed": seed}
            for arm in ("A", "B", "D"):
                genes = run[arm].get("final_genes") or None
                genome = seed_genome(f"study-{arm}", evolvable=set(), overrides=genes)
                firsts, finals, passes, false_pass = [], [], 0, 0
                for task in heldout:
                    tr = run_task(genome, task)
                    first = next(s for s in tr.steps if s.step_id == "generate-0").output_summary
                    crit = next(s for s in tr.steps if s.step_id == "critique-0").output_summary
                    ok = mc_pass(first or "", task)
                    firsts.append(1.0 if ok else 0.0)
                    finals.append(float(tr.metadata["eval_score"]))
                    said = (crit or "").strip().upper().startswith("PASS")
                    passes += said
                    false_pass += said and not ok
                final = st.mean(finals)
                if abs(final - run[arm]["opt_heldout"]) > 1e-9:
                    raise AssertionError(
                        f"{level} seed {seed} arm {arm}: replayed {final} != released "
                        f"{run[arm]['opt_heldout']}; the replay is not the run"
                    )
                row[arm] = {
                    "p0": st.mean(firsts),
                    "final": final,
                    "pass_rate": passes / len(heldout),
                    "false_pass": false_pass,
                    "genes_changed": run[arm].get("genes_changed") or [],
                }
            rows.append(row)
    return {"rows": rows}


def main() -> int:  # pragma: no cover - CLI convenience
    data = {lv: _replay(lv) for lv in LEVELS}
    report: dict[str, Any] = {"levels": {}}
    deltas = {}
    for lv, v in data.items():
        rows = v["rows"]
        d = [r["D"]["final"] - r["B"]["final"] for r in rows]
        deltas[lv] = d
        a_mean = st.mean(r["A"]["final"] for r in rows)
        report["levels"][lv] = {
            "n_seeds": len(rows),
            "M1_realized_p0": st.mean(r["A"]["p0"] for r in rows),
            "M2_armA_heldout": a_mean,
            "M2_saturated": a_mean > SATURATION,
            "M3_seeds_D_changed_verify": sum(VERIFY_GENE in r["D"]["genes_changed"] for r in rows),
            "B_mean": st.mean(r["B"]["final"] for r in rows),
            "D_mean": st.mean(r["D"]["final"] for r in rows),
            "delta": pooled(d),
            "delta_per_seed": d,
            "D_pass_rate": st.mean(r["D"]["pass_rate"] for r in rows),
            "D_seeds_pass_gt0": sum(r["D"]["pass_rate"] > 0 for r in rows),
            "D_false_pass_pooled": sum(r["D"]["false_pass"] for r in rows),
            "A_false_pass_pooled": sum(r["A"]["false_pass"] for r in rows),
        }
        # Added after the results were read, so labelled exploratory: the same first-critique
        # counts for the seed critique (A) and the prompt-only arm (B), which say whether D's
        # PASS rate moved because the critique evolved or was already there in the seed.
        report.setdefault("exploratory_by_arm", {})[lv] = {
            arm: {
                "realized_p0": st.mean(r[arm]["p0"] for r in rows),
                "pass_rate": st.mean(r[arm]["pass_rate"] for r in rows),
                "seeds_pass_gt0": sum(r[arm]["pass_rate"] > 0 for r in rows),
                "false_pass_pooled": sum(r[arm]["false_pass"] for r in rows),
            }
            for arm in ("A", "B", "D")
        }
    s = swing(deltas["L2"], deltas["L1"])
    t, df = welch(deltas["L2"], deltas["L1"])
    report["H1"] = {
        **s,
        "welch_t": t,
        "welch_df": df,
        "welch_p": welch_p(t, df),
        "verdict": verdict(s["S"], s["ci"]),
        "M1_ordered": report["levels"]["L2"]["M1_realized_p0"]
        > report["levels"]["L1"]["M1_realized_p0"],
    }
    out = Path("docs/qwen-l1l2/analysis.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
