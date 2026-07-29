#!/usr/bin/env python
"""Check every machine-checkable number in the paper against the released artifacts.

Zero cost, offline. The paper quotes dozens of numbers derived from the committed results
JSONs and cassettes; each was correct when written, and each is one careless edit away from
no longer being. This script re-derives them from the artifacts and fails loudly on any
drift, so "the paper agrees with its data" is a test rather than a memory.

It exists because a fact-check pass found two real errors that had survived several
readings: the paper said three runs overrode the \\$5 default when four did, and it said
the reproducibility script registers eleven runs after two more had been added. Both are
the kind of quiet arithmetic drift that a reviewer finds and an author does not.

A later pass added the Evidence-8 checks, which exist for a sharper reason: the ad-hoc
integrity scan that should have caught an empty reflection completion was itself vacuous
(it serialised each cassette entry to JSON before measuring length, so it compared the size
of a dictionary and could never see an empty output). Those numbers are pinned here so the
check has to keep working rather than merely having worked once.

A third pass added the `tex:` checks, for the same reason one level up. Every other check
compares an artifact against a number typed into *this file*, so it verifies the data and
not the paper; the two can disagree. They did: the closability table claimed the corrected
re-run changed `ensure_correct` on 2/3 seeds while the data and the body text both say 3/3
(the 2/3 is arm B's `generate`, transposed). The `tex:` checks scrape main.tex itself, so
the claimed side is the paper's own sentence. A pattern that stops matching yields NaN and
fails, because a reworded sentence must break the check loudly rather than quietly stop
being checked.

Scope: numbers recomputable from `docs/*.json`, the vendored task data, and the committed
cassettes, plus the headline numbers as they are written in `docs/paper/main.tex`. Claims
that are not machine-checkable (unquantified prose, cited literature, one-off replays
against superseded commits) are out of scope and listed at the bottom of the output so the
gap is visible rather than implied.

    python scripts/verify_paper_claims.py
"""

from __future__ import annotations

import json
import re
from statistics import mean, pstdev

DOCS = "docs"
PAPER = f"{DOCS}/paper/main.tex"


def _runs(name: str) -> list[dict]:
    with open(f"{DOCS}/{name}", encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _tex() -> str:
    with open(PAPER, encoding="utf-8") as fh:
        return fh.read()


def _scrape(tex: str, pattern: str) -> float:
    """Pull one number out of the paper's own prose.

    Returns NaN when the pattern does not match, which fails the check rather than
    skipping it. That direction matters: a reworded sentence must break the check loudly,
    because a check that silently stops looking is exactly the failure this script exists
    to prevent (see the vacuous integrity scan in the module docstring).
    """
    m = re.search(pattern, tex, re.DOTALL)
    if m is None:
        return float("nan")
    return float(m.group(1))


def _arm_mean(name: str, arm: str, field: str) -> float:
    return mean(r[arm][field] for r in _runs(name))


#: (claim, expected, tolerance, thunk). Tolerance matches the precision the paper prints at.
def _checks() -> list[tuple[str, float, float, float]]:
    out: list[tuple[str, float, float, float]] = []

    # --- Table "detector quality vs loop performance" (the non-monotonicity claim) -----
    for label, f, claimed in (
        ("tab:curve run3 intrinsic seed", "validation-study-multihard.json", 0.765),
        ("tab:curve run4 tool-fed seed", "validation-study-feedback.json", 0.837),
        ("tab:curve run7 oracle mid r1", "validation-study-oracle.json", 0.748),
        ("tab:curve run8 oracle mid r3", "validation-study-fixbudget.json", 0.813),
        ("tab:curve run9 oracle lo r3", "validation-study-fbactlo.json", 0.810),
        ("tab:curve run10 oracle hi r3", "validation-study-fbacthi.json", 0.873),
    ):
        out.append((label, claimed, 0.0005, _arm_mean(f, "A", "seed_heldout")))

    # --- Evidence 3b: the per-seed actuator decomposition -----------------------------
    act = _runs("validation-study-bcb-actuator-big.json")
    for i, (bs, bo, ds, do) in enumerate(
        [(0.55, 0.60, 0.55, 0.45), (0.45, 0.60, 0.45, 0.45), (0.65, 0.65, 0.65, 0.65)]
    ):
        for arm, cs, co in (("B", bs, bo), ("D", ds, do)):
            out.append((f"ev3b seed{i} {arm} seed", cs, 0.005, act[i][arm]["seed_heldout"]))
            out.append((f"ev3b seed{i} {arm} opt", co, 0.005, act[i][arm]["opt_heldout"]))

    # --- The worker-swap replication (run 11) -----------------------------------------
    nano = _runs("validation-study-multihard-nano.json")
    out.append(("run11 arm B mean", 0.70, 0.005, mean(r["B"]["opt_heldout"] for r in nano)))
    out.append(("run11 arm B std", 0.03, 0.005, pstdev([r["B"]["opt_heldout"] for r in nano])))
    out.append(("run11 arm D mean", 0.69, 0.005, mean(r["D"]["opt_heldout"] for r in nano)))
    out.append(("run11 arm D std", 0.03, 0.005, pstdev([r["D"]["opt_heldout"] for r in nano])))
    out.append(
        (
            "run11 seed",
            0.67,
            0.005,
            _arm_mean("validation-study-multihard-nano.json", "A", "seed_heldout"),
        )
    )

    # --- Headline effects and intervals -----------------------------------------------
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_c1 import paired_by_task
    from analyze_deltas import delta_arm, delta_between, exact_resample_ci, sign_test_p

    d_act = delta_arm(f"{DOCS}/validation-study-bcb-actuator-big.json")
    m, lo, hi = exact_resample_ci(d_act)
    out.append(("actuator delta(D-B)", -0.10, 0.005, m))
    out.append(("actuator CI lo", -0.150, 0.0005, lo))
    out.append(("actuator CI hi", 0.000, 0.0005, hi))
    out.append(("actuator sign p", 0.50, 0.005, sign_test_p(d_act)))

    d_hi = delta_between(
        f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(d_hi)
    out.append(("actionability delta(hi-lo)", 0.063, 0.0005, m))
    out.append(("actionability CI lo", 0.025, 0.0005, lo))
    out.append(("actionability CI hi", 0.090, 0.0005, hi))
    out.append(("actionability sign p", 0.25, 0.005, sign_test_p(d_hi)))

    d_code = delta_between(
        f"{DOCS}/validation-study-bcb-acthi.json", f"{DOCS}/validation-study-bcb-actlo.json"
    )
    out.append(("code actionability delta", 0.200, 0.0005, exact_resample_ci(d_code)[0]))

    d_n3 = delta_arm(f"{DOCS}/validation-study-multihard.json")
    out.append(("run3 delta(D-B)", 0.00, 0.0005, exact_resample_ci(d_n3)[0]))
    d_n11 = delta_arm(f"{DOCS}/validation-study-multihard-nano.json")
    out.append(("run11 delta(D-B)", -0.014, 0.0005, exact_resample_ci(d_n11)[0]))

    # --- The C1 verifier ablation (Analysis 1d) ---------------------------------------
    c1 = list(
        paired_by_task(
            f"{DOCS}/validation-study-c1-base.json", f"{DOCS}/validation-study-c1-hand.json"
        ).values()
    )
    out.append(("C1 n tasks", 32, 0, len(c1)))
    out.append(("C1 ties", 17, 0, sum(1 for d in c1 if d == 0.0)))
    out.append(("C1 hand better", 6, 0, sum(1 for d in c1 if d > 0)))
    out.append(("C1 seed better", 9, 0, sum(1 for d in c1 if d < 0)))
    out.append(("C1 mean delta", -0.033, 0.0005, mean(c1)))
    out.append(("C1 sign p", 0.61, 0.005, sign_test_p(c1)))

    # --- The corrected-harness re-run, C2 (tab:rerun and the resolved prediction) ------
    # The gene-change counts are the load-bearing ones: they are what distinguishes this
    # run from the eleven it replaces, and the claim "the treatment was applied" reduces
    # entirely to them. Checked from the released JSON rather than from the prose.
    c2 = _runs("validation-study-c2.json")
    out.append(("C2 seeds", 3, 0, len(c2)))
    for arm, seed_m, opt_m in (
        ("A", 0.790, 0.790),
        ("C", 0.790, 0.790),
        ("B", 0.790, 0.750),
        ("D", 0.790, 0.735),
    ):
        out.append(
            (f"C2 {arm} seed mean", seed_m, 0.0005, mean(r[arm]["seed_heldout"] for r in c2))
        )
        out.append((f"C2 {arm} opt mean", opt_m, 0.0005, mean(r[arm]["opt_heldout"] for r in c2)))
    out.append(
        (
            "C2 arm D evolved ensure_correct on N seeds",
            2,
            0,
            sum(1 for r in c2 if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        )
    )
    out.append(
        (
            "C2 arm C changed nothing on N seeds",
            3,
            0,
            sum(1 for r in c2 if not (r["C"].get("genes_changed") or [])),
        )
    )
    d_c2 = delta_arm(f"{DOCS}/validation-study-c2.json")
    m, lo, hi = exact_resample_ci(d_c2)
    out.append(("C2 delta(D-B)", -0.015, 0.0005, m))
    out.append(("C2 CI lo", -0.065, 0.0005, lo))
    out.append(("C2 CI hi", 0.020, 0.0005, hi))
    # The per-seed drops the overfitting mechanism is stated on: seed 2 is where arm D
    # accepted two candidates against arm B's one and fell furthest, and seed 0 is where
    # no arm accepted anything. The acceptance counts themselves come from the run log
    # and are NOT in the released JSON, so only the scores they explain are checked here.
    # Evidence 8: the seed-2 arm-D critique gene is EMPTY, and two reflection completions
    # were empty. Pinned because the paper's "the treatment was applied" claim is qualified
    # entirely by these two numbers, and because the check that should have caught it
    # (scanning the cassette for empty outputs) was itself vacuous until it was rewritten.
    out.append(
        ("C2 seed2 D ensure_correct length", 0, 0, len(c2[2]["D"]["final_genes"]["ensure_correct"]))
    )
    out.append(
        (
            "C2 seed1 D ensure_correct length",
            3101,
            0,
            len(c2[1]["D"]["final_genes"]["ensure_correct"]),
        )
    )
    with open("tests/fixtures/cassettes/study-c2.json", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]
    vals = list(entries.values()) if isinstance(entries, dict) else entries
    refl = [v["output"] for v in vals if v["name"] == "reflection"]
    out.append(("C2 empty reflection completions", 2, 0, sum(1 for o in refl if not o.strip())))
    out.append(
        (
            "C2 empty worker completions",
            0,
            0,
            sum(1 for v in vals if v["name"] == "worker" and not v["output"].strip()),
        )
    )
    out.append(("C2 seed2 B opt", 0.750, 0.0005, c2[2]["B"]["opt_heldout"]))
    out.append(("C2 seed2 D opt", 0.685, 0.0005, c2[2]["D"]["opt_heldout"]))

    # --- C2b: the CLEAN re-run, which is what tab:rerun and the conclusion report --------
    # C2 above is retained only as the evidence for Evidence 8. Everything the paper claims
    # about the corrected harness -- treatment applied on 3/3 seeds, prompt-only flat,
    # loop-level costing 0.042 -- is checked here against the clean run.
    c2b = _runs("validation-study-c2b.json")
    out.append(("C2b seeds", 3, 0, len(c2b)))
    for arm, opt_m in (("A", 0.788), ("C", 0.788), ("B", 0.788), ("D", 0.747)):
        out.append(
            (f"C2b {arm} seed mean", 0.788, 0.0005, mean(r[arm]["seed_heldout"] for r in c2b))
        )
        out.append((f"C2b {arm} opt mean", opt_m, 0.0005, mean(r[arm]["opt_heldout"] for r in c2b)))
    out.append(
        (
            "C2b arm D evolved ensure_correct on N seeds",
            3,
            0,
            sum(1 for r in c2b if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        )
    )
    out.append(
        (
            "C2b arm C changed nothing on N seeds",
            3,
            0,
            sum(1 for r in c2b if not (r["C"].get("genes_changed") or [])),
        )
    )
    # The evolved rubrics are substantial, against a 127-char seed -- the contrast with C2's
    # zero-length gene is the whole point of the re-run.
    for i, n in enumerate((3235, 3022, 2920)):
        out.append(
            (
                f"C2b seed{i} D ensure_correct length",
                n,
                0,
                len(c2b[i]["D"]["final_genes"]["ensure_correct"]),
            )
        )
    out.append(("C2b seed gene length", 127, 0, len(c2b[0]["C"]["final_genes"]["ensure_correct"])))
    d_c2b = delta_arm(f"{DOCS}/validation-study-c2b.json")
    m, lo, hi = exact_resample_ci(d_c2b)
    out.append(("C2b delta(D-B)", -0.042, 0.0005, m))
    out.append(("C2b CI lo", -0.095, 0.0005, lo))
    out.append(("C2b CI hi", 0.050, 0.0005, hi))
    out.append(("C2b sign p", 1.00, 0.005, sign_test_p(d_c2b)))
    # No empty completion anywhere: the defect Evidence 8 describes did not recur.
    with open("tests/fixtures/cassettes/study-c2b.json", encoding="utf-8") as fh:
        b_entries = json.load(fh)["entries"]
    b_vals = list(b_entries.values()) if isinstance(b_entries, dict) else b_entries
    out.append(("C2b empty completions", 0, 0, sum(1 for v in b_vals if not v["output"].strip())))
    # The evolved critique scored as a detector: PASS emitted zero times on every seed, and
    # pooled true positives 0 against the seed rubric's 1. This is the paper's mechanism for
    # arm D's loss, so it is checked rather than remembered. Replaying is slow (three
    # cassette passes), so only the two load-bearing integers are pinned here; the full
    # table comes from scripts/analyze_evolved_gene.py.
    out.append(
        (
            "C2b evolved rubric lengths all > 2500",
            3,
            0,
            sum(1 for r in c2b if len(r["D"]["final_genes"]["ensure_correct"]) > 2500),
        )
    )
    out.append(
        (
            "C2b shortest reflection output",
            968,
            0,
            min(len(v["output"]) for v in b_vals if v["name"] == "reflection"),
        )
    )

    # --- The metric's do-nothing floor (Threats) --------------------------------------
    from whystuck.eval.ifbench import ifbench_score, load_ifbench
    from whystuck.eval.multi_constraint import load_multi_constraint, mc_score

    ifb = load_ifbench("all")
    mh = load_multi_constraint("all", pool=64, hard=True)
    out.append(("floor ifbench empty", 0.548, 0.0005, mean([ifbench_score("", t) for t in ifb])))
    out.append(("floor ifbench 'x'", 0.452, 0.0005, mean([ifbench_score("x", t) for t in ifb])))
    out.append(("floor multihard empty", 0.420, 0.0005, mean([mc_score("", t) for t in mh])))
    out.append(("floor multihard 'x'", 0.499, 0.0005, mean([mc_score("x", t) for t in mh])))
    out.append(("ifbench pool size", 62, 0, len(ifb)))

    # --- The oracle taxonomy's verdict-equivalence count ------------------------------
    verdict_equivalent = {"exact_word_count", "word_max_length", "end_with", "num_bullets"}
    inst = [i for t in mh for i in t.payload["instruction_id_list"]]
    out.append(("taxonomy instances total", 288, 0, len(inst)))
    out.append(
        ("taxonomy verdict-equivalent", 165, 0, sum(1 for i in inst if i in verdict_equivalent))
    )
    out.append(("taxonomy families total", 8, 0, len(set(inst))))
    out.append(("taxonomy hard-pool tasks", 64, 0, len(mh)))

    # --- Cost overrides: how many runs raised --max-usd above the $5 default ----------
    import glob

    est = []
    for f in glob.glob(f"{DOCS}/validation-study*.json"):
        with open(f, encoding="utf-8") as fh:
            est.append(json.load(fh)["meta"].get("estimated_usd", 0))
    out.append(("runs over the $5 default", 4, 0, sum(1 for u in est if u > 5.0)))
    out.append(("largest estimate", 11.76, 0.005, max(est)))

    # Prose cross-checks. Everything above re-derives a number from the artifacts and
    # compares it to a number typed into *this file*; nothing above reads the paper. That
    # gap let a real error through: the closability table said the re-run changed
    # ensure_correct on 2/3 seeds when the data (and the body text) say 3/3 -- the 2/3 is
    # arm B's generate, transposed. The checks below scrape the paper's own sentences, so
    # the claimed side comes from main.tex rather than from an author's memory of it.
    tex = _tex()
    d_row = r"D & \\texttt\{generate\} \$\+\$ \\texttt\{ensure\\_correct\} & 0\.788 & "
    prose: list[tuple[str, str, float, float]] = [
        (
            "tex: D evolved ensure_correct on N/3",
            r"corrected harness changed \\texttt\{ensure\\_correct\} on \$(\d)/3\$",
            0,
            sum(1 for r in c2b if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        ),
        (
            "tex: tab:rerun D optimized",
            d_row + r"([\d.]+) &",
            0.0005,
            mean(r["D"]["opt_heldout"] for r in c2b),
        ),
        (
            "tex: tab:rerun D delta",
            d_row + r"[\d.]+ & \$\\mathbf\{(-?[\d.]+)\}",
            0.0005,
            mean(r["D"]["opt_heldout"] - r["D"]["seed_heldout"] for r in c2b),
        ),
        (
            "tex: shortest reflection output",
            r"shortest reflection output: \$(\d+)\$",
            0,
            968,
        ),
        (
            "tex: seed rubric length",
            r"against a \$(\d+)\$-character seed",
            0,
            len(c2b[0]["C"]["final_genes"]["ensure_correct"]),
        ),
    ]
    for i in range(3):
        prose.append(
            (
                f"tex: evolved rubric {i} length",
                r"detector rubrics of " + r"\$\d+\$,?\s*(?:and\s*)?" * i + r"\$(\d+)\$",
                0,
                len(c2b[i]["D"]["final_genes"]["ensure_correct"]),
            )
        )
        prose.append(
            (
                f"tex: per-seed delta(D-B) {i}",
                r"Per-seed \$\\dd\(D-B\) = \[" + r"-?[\d.+]+, " * i + r"(-?[\d.+]+)",
                0.0005,
                d_c2b[i],
            )
        )
    for label, pattern, tol, artifact in prose:
        out.append((label, _scrape(tex, pattern), tol, artifact))

    return out


def main() -> int:  # pragma: no cover - CLI convenience
    failures = []
    checks = _checks()
    for label, claimed, tol, actual in checks:
        ok = abs(actual - claimed) <= tol
        if not ok:
            failures.append((label, claimed, actual))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:32} paper={claimed:<9} artifact={actual}")
    print(
        f"\n{len(checks) - len(failures)}/{len(checks)} paper claims match the released artifacts."
    )
    if failures:
        print("\nDRIFT:")
        for label, claimed, actual in failures:
            print(f"  {label}: paper says {claimed}, artifacts say {actual}")
        return 1
    print(
        "\nNOT machine-checkable here (verified by hand, listed so the gap is visible):\n"
        "  - Evidence 1's per-arm gene-change counts: they replay only against the pre-fix\n"
        "    commit, since the corrected reflective dataset changes every cassette key.\n"
        "    Re-verified 2026-07-28 at fe670a8: generate 5/15, ensure_correct 0/15, and\n"
        "    per arm B 3/5, C 0/5, D 2/5 generate with 0/5 ensure_correct.\n"
        "  - Evidence 7's 70 swallowed misses and the 0.666 -> 0.681 substitution, which\n"
        "    require a deliberately mismatched replay.\n"
        "  - The stopped mmc=160 run's partial spend, whose cassette is not released.\n"
        "  - Every cited external result, and every prose claim that is not one of the\n"
        "    `tex:` numbers above -- those are scraped from main.tex, the rest are not."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
