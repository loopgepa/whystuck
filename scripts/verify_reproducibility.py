#!/usr/bin/env python
"""Prove every paid run reproduces bit-for-bit from its cassette (offline, $0).

For each of the seven pre-registered runs, this re-runs the FULL pipeline (loaders
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
from pathlib import Path

import run_validation_study as cli

CASSETTES = "tests/fixtures/cassettes"

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
]


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
    all_ok = True
    for label, committed, cassette, args in RUNS:
        try:
            ok, detail = verify_one(committed, cassette, args)
        except Exception as exc:  # a cassette miss (nondeterminism) surfaces here
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        all_ok &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {label:22} {detail}")
    print(
        "\n"
        + (
            "ALL 7 RUNS REPRODUCE FROM CASSETTE ($0, offline)."
            if all_ok
            else "REPRODUCTION FAILED."
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
