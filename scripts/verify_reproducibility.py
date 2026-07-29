#!/usr/bin/env python
"""Replay the SUPERSEDED paid runs from their cassettes (offline, $0).

STATUS — read this before quoting any number this script prints. The eleven runs listed
below were recorded under a harness with three defects that a later audit established:

  1. the reflective dataset was component-agnostic (`adapter.py` handed every gene the
     GENERATOR's input/output), so the critique gene was optimized blind to the critique;
  2. GEPA ran under its default one-component-per-proposal selector, so a two-gene arm
     searched the productive gene about half as hard as the one-gene arm; and
  3. arm C's control was saturated (a perfect 1.0), so it proposed nothing and could not
     detect the failure it advertised.

The harness has since been corrected, and the corrected wiring changes every reflection
prompt — hence every cassette key. These runs therefore CANNOT be replayed by the current
code, and this script no longer asserts that they do. It is retained to document what was
recorded, and to replay any run whose cassette was captured under the current wiring; runs
in `SUPERSEDED` are reported and skipped rather than silently "verified".

`Δ(D−B)` from a superseded run is not evidence about loop-level optimization: on the text
runs the verify gene was byte-identical to the seed in every arm-run inspected, so the
treatment was never applied and the zero is an identity. Re-run under the corrected
harness before drawing conclusions.

Run with the eval venv (has gepa+litellm):
    scratchpad/evalvenv/bin/python scripts/verify_reproducibility.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import run_validation_study as cli

CASSETTES = "tests/fixtures/cassettes"
# Every entry below predates the harness correction, so none of them can replay under the
# current code. Listing them explicitly (rather than deleting them) keeps the record of
# what was run, and makes the count of still-verifiable runs honest: zero, until a run is
# recorded under the corrected wiring.
SUPERSEDED = frozenset(
    {
        "run1 single-constraint",
        "run2 easy-multi",
        "run3 hard-multi",
        "run4 feedback",
        "run5 plan",
        "run6 cross-lineage",
        "run7 oracle",
        "run8 fix-budget",
        "run9 fb-actionability lo",
        "run10 fb-actionability hi",
        "D5-code MBPP+ (saturated)",
        # Not superseded by the wiring correction like the eleven above, but by the
        # blank-gene guard the run itself motivated. See EXPECTED_RUNS below.
        "C2 four-arm corrected harness",
    }
)
EXPECTED_RUNS = 5  # corrected harness: two C1 conditions, the C2b re-run, and p0 L2/L3.
# C2 is NOT among them. Auditing it found that two reflection completions came back empty and
# GEPA accepted one as the verify gene, so `genome_from_candidate` now refuses blank gene text
# -- and that guard makes C2's own cassette unreplayable, because replaying it re-serves the
# empty completion and the guard (correctly) raises. Its numbers stand as measured but are not
# verifiable under HEAD, which is exactly the state SUPERSEDED exists to record. C2b is the
# re-run under the guard and the raised reflection cap, and it does replay bit-for-bit.

# (label, committed results JSON, cassette, exact CLI args). reflection_calls and
# task-set come from each run's pre-registration (not stored in the older metas).
RUNS = [
    # --- recorded under the CORRECTED harness (not superseded) --------------------
    # C2b: the four-arm re-run the retraction called for, and the run the paper's
    # "the treatment was applied" claim rests on -- arm D changed `ensure_correct` on 3 of 3
    # seeds, against 0 of 15 before the fix, with no empty completions anywhere. It replaces
    # C2 (below, superseded), whose seed 2 was corrupted by the empty-completion defect that
    # motivated both the blank-gene guard and the raised reflection token cap.
    (
        "C2b four-arm corrected harness",
        "docs/validation-study-c2b.json",
        f"{CASSETTES}/study-c2b.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "ABCD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
            "--module-selector",
            "all",
            "--reflection-model",
            "openai/gpt-5.6-luna",
        ],
    ),
    # The p0 sweep: the runs that turned the null into a conditional result. L1 IS the
    # C2b row above, so only the two new levels are registered here.
    (
        "p0 sweep L2 (p0=0.267)",
        "docs/validation-study-p0-L2.json",
        f"{CASSETTES}/study-p0-L2.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "BD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
            "--module-selector",
            "all",
            "--reflection-model",
            "openai/gpt-5.6-luna",
            "--exclude-constraint",
            "exact_word_count",
        ],
    ),
    (
        "p0 sweep L3 (p0=0.667)",
        "docs/validation-study-p0-L3.json",
        f"{CASSETTES}/study-p0-L3.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "BD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
            "--module-selector",
            "all",
            "--reflection-model",
            "openai/gpt-5.6-luna",
            "--exclude-constraint",
            "exact_word_count",
            "--exclude-constraint",
            "word_max_length",
        ],
    ),
    (
        "C2 four-arm corrected harness",
        "docs/validation-study-c2.json",
        f"{CASSETTES}/study-c2.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "ABCD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
            "--module-selector",
            "all",
            "--reflection-model",
            "openai/gpt-5.6-luna",
        ],
    ),
    # The C1 verifier ablation: two conditions differing ONLY in the ensure_correct
    # gene, sharing one content-addressed cassette so the first draft per task is
    # identical across them. These are the first runs whose replay proves anything
    # about the harness the paper now describes.
    (
        "C1 verifier ablation (seed rubric)",
        "docs/validation-study-c1-base.json",
        f"{CASSETTES}/study-c1.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "A",
            "--seeds",
            "0",
            "--n-train",
            "32",
            "--n-heldout",
            "32",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "C1 verifier ablation (hand rubric)",
        "docs/validation-study-c1-hand.json",
        f"{CASSETTES}/study-c1.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "A",
            "--seeds",
            "0",
            "--n-train",
            "32",
            "--n-heldout",
            "32",
            "--loop-max-retries",
            "1",
            "--seed-gene",
            "ensure_correct=docs/genes/verify_measure_v2.txt",
        ],
    ),
    # --- superseded: recorded under the pre-correction harness --------------------
    (
        "run1 single-constraint",
        "docs/validation-study.json",
        f"{CASSETTES}/study.json",
        [
            "--task-set",
            "ifbench",
            "--seeds",
            "0,1",
            "--n-train",
            "16",
            "--n-heldout",
            "16",
            "--max-metric-calls",
            "64",
            "--reflection-calls",
            "10",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run2 easy-multi",
        "docs/validation-study-multi.json",
        f"{CASSETTES}/study-multi.json",
        [
            "--task-set",
            "multi",
            "--seeds",
            "0,1,2,3,4",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run3 hard-multi",
        "docs/validation-study-multihard.json",
        f"{CASSETTES}/study-multihard.json",
        [
            "--task-set",
            "multi-hard",
            "--seeds",
            "0,1,2,3,4",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "28",
            "--reflection-calls",
            "6",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run4 feedback",
        "docs/validation-study-feedback.json",
        f"{CASSETTES}/study-feedback.json",
        [
            "--task-set",
            "multi-hard",
            "--feedback-stats",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run5 plan",
        "docs/validation-study-plan.json",
        f"{CASSETTES}/study-plan.json",
        [
            "--task-set",
            "multi-hard",
            "--plan-gene",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run6 cross-lineage",
        "docs/validation-study-crossfamily.json",
        f"{CASSETTES}/study-crossfamily.json",
        [
            "--task-set",
            "multi-hard",
            "--critique-model",
            "openai/gpt-5.6-luna",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run7 oracle",
        "docs/validation-study-oracle.json",
        f"{CASSETTES}/study-oracle.json",
        [
            "--task-set",
            "multi-hard",
            "--oracle-critique",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "1",
        ],
    ),
    (
        "run8 fix-budget",
        "docs/validation-study-fixbudget.json",
        f"{CASSETTES}/study-fixbudget.json",
        [
            "--task-set",
            "multi-hard",
            "--oracle-critique",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "3",
        ],
    ),
    (
        "run9 fb-actionability lo",
        "docs/validation-study-fbactlo.json",
        f"{CASSETTES}/study-fbactlo.json",
        [
            "--task-set",
            "multi-hard",
            "--oracle-critique",
            "--feedback-level",
            "lo",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "3",
        ],
    ),
    (
        "run10 fb-actionability hi",
        "docs/validation-study-fbacthi.json",
        f"{CASSETTES}/study-fbacthi.json",
        [
            "--task-set",
            "multi-hard",
            "--oracle-critique",
            "--feedback-level",
            "hi",
            "--seeds",
            "0,1,2",
            "--n-train",
            "8",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "3",
        ],
    ),
    (
        # Cross-domain (appendix): MBPP+ saturated its train split (all arms == seed,
        # a degenerate zero), but it still reproduces bit-for-bit — pure-function tasks
        # execute deterministically in the subprocess sandbox, so replay is exact.
        "D5-code MBPP+ (saturated)",
        "docs/validation-study-code.json",
        f"{CASSETTES}/study-code.json",
        [
            "--task-set",
            "code",
            "--seeds",
            "0,1",
            "--n-train",
            "6",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "4",
            "--loop-max-retries",
            "1",
        ],
    ),
]

# The cross-domain BigCodeBench-Hard run (D5, docs/validation-study-bcb.json) is
# INTENTIONALLY NOT verified here: it is not bit-for-bit replayable. Model solutions may
# use unseeded randomness, so re-executing them in the Docker sandbox yields run-to-run
# variation that shifts the content-addressed prompt key — an intrinsic property of
# executable-code benchmarks. Per-execution grading is deterministic and the headline
# reproduced identically across two independent paid runs (see the paper, "Reproducibility
# scope"); strict LLM-call replay is out of scope for it by construction, not oversight.


def _scores(runs: list[dict]) -> list[tuple]:
    """Every (seed, arm) held-out pair, over the arms each seed ACTUALLY carries.

    Sorted so the comparison never depends on dict ordering. Iterating a hardcoded
    ("A","B","C","D") raised KeyError on a single-arm run, which the ablation runs are:
    C1 evolves nothing and needs only arm A. Comparing the arms present keeps the check
    strict -- a replay that dropped an arm changes the key set and still fails -- while
    letting a run register without paying for arms its question does not use."""
    return [
        (i, arm, r[arm]["seed_heldout"], r[arm]["opt_heldout"])
        for i, r in enumerate(runs)
        for arm in sorted(r)
    ]


def verify_one(committed: str, cassette: str, args: list[str]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "replay.json")
        code = cli.main(
            [
                "--replay",
                *args,
                "--record",
                cassette,
                "--out",
                out,
                "--report",
                str(Path(d) / "r.md"),
                "--max-usd",
                "99",
            ]
        )
        if code != 0:
            return False, f"exit {code}"
        want = json.loads(Path(committed).read_text())["runs"]
        got = json.loads(Path(out).read_text())["runs"]
    if _scores(want) == _scores(got):
        return True, "exact match"
    return False, "score mismatch"


def main() -> int:
    verified = 0
    all_ok = True
    skipped = 0
    for label, committed, cassette, args in RUNS:
        if label in SUPERSEDED:
            skipped += 1
            print(f"[SKIP] {label:22} recorded under the pre-correction harness")
            continue
        try:
            ok, detail = verify_one(committed, cassette, args)
        except Exception:  # noqa: BLE001 — a cassette miss OR a real code regression
            # Keep the traceback so a genuine bug is not misread as "just nondeterminism".
            ok, detail = False, traceback.format_exc().strip().splitlines()[-1]
        verified += 1
        all_ok &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {label:22} {detail}")
    print(f"\n{skipped} run(s) skipped as superseded; {verified} replayable run(s) checked.")
    # Guard against a vacuous pass: an empty/truncated RUNS list, or a list where every
    # entry is superseded, must NOT print success. `all_ok` seeds True, so without the
    # `verified > 0` term a zero-iteration loop would claim reproduction — which is exactly
    # the failure mode this script exists to prevent.
    complete = all_ok and verified > 0 and verified >= EXPECTED_RUNS
    if verified == 0:
        # Say this plainly rather than exiting 0 on an empty check, which would read as
        # "reproduction proven" when nothing was actually replayed.
        print(
            "NOTHING PROVEN: no run has yet been recorded under the corrected harness. "
            "Re-run the study, then register it here."
        )
    else:
        print(
            f"ALL {verified} RUNS REPRODUCE FROM CASSETTE ($0, offline)."
            if complete
            else "REPRODUCTION FAILED."
        )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
