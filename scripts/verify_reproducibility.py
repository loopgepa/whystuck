#!/usr/bin/env python
"""Prove every paid run reproduces bit-for-bit from its cassette (offline, $0).

For each of the eleven bit-for-bit runs (ten IFBench + the MBPP+ cross-domain run;
BigCodeBench-Hard is excluded by construction, see the note after RUNS), this re-runs
the FULL pipeline (loaders
+ GEPA + scoring) in strict `--replay` mode against the committed cassette — no API
key, no spend, and any cassette miss raises — then checks the replayed per-arm
seed/optimized held-out scores equal the committed results JSON exactly. GEPA is
deterministic on replay (verified), so a clean match end-to-end demonstrates full
reproducibility.

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
EXPECTED_RUNS = 11  # 10 IFBench runs + the MBPP+ cross-domain run; all bit-for-bit.
# (BigCodeBench-Hard is excluded by construction — see the note after RUNS.)

# (label, committed results JSON, cassette, exact CLI args). reflection_calls and
# task-set come from each run's pre-registration (not stored in the older metas).
RUNS = [
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
    return [
        (i, arm, r[arm]["seed_heldout"], r[arm]["opt_heldout"])
        for i, r in enumerate(runs)
        for arm in ("A", "B", "C", "D")
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
    for label, committed, cassette, args in RUNS:
        try:
            ok, detail = verify_one(committed, cassette, args)
        except Exception:  # noqa: BLE001 — a cassette miss OR a real code regression
            # Keep the traceback so a genuine bug is not misread as "just nondeterminism".
            ok, detail = False, traceback.format_exc().strip().splitlines()[-1]
        verified += 1
        all_ok &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {label:22} {detail}")
    # Guard against a vacuous pass: an empty/truncated RUNS list must NOT print success.
    # `all_ok` seeds True, so without this a zero-iteration loop would claim reproduction.
    complete = all_ok and verified >= EXPECTED_RUNS
    if all_ok and verified < EXPECTED_RUNS:
        print(f"\nINCOMPLETE: only {verified}/{EXPECTED_RUNS} runs registered — NOT proven.")
    else:
        print(
            f"\nALL {verified} RUNS REPRODUCE FROM CASSETTE ($0, offline)."
            if complete
            else "REPRODUCTION FAILED."
        )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
