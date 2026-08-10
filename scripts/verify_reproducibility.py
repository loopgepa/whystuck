#!/usr/bin/env python
"""Replay every released paid run from its cassette (offline, $0), or say why it cannot.

Each released run reaches exactly one of three states, and the point of the script is that
none of them is silence: it REPLAYS bit-for-bit, or it is registered with frozen args and
SUPERSEDED with a stated reason, or it is listed in `UNREPLAYABLE` with a stated reason.
`tests/test_verify_reproducibility.py` asserts the partition is total, so a run cannot back a
published number while quietly holding no replay status at all.

STATUS — read this before quoting any number this script prints. Eleven of the runs listed
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
#: label -> why this registered run cannot replay under HEAD. A dict rather than a set because
#: the reasons are NOT the same, and printing one reason for all of them was itself a small
#: instance of this study's subject: for eleven entries "recorded under the pre-correction
#: harness" is true, and for `C2` it is false -- C2 was recorded under the corrected wiring and
#: is unreplayable for an unrelated reason (the blank-gene guard it motivated). The skip line
#: now states each entry's own reason, so a reader cannot be told a true-sounding falsehood
#: about why a number is not verifiable.
_WIRING = "recorded under the pre-correction harness (component-agnostic reflective dataset)"
SUPERSEDED: dict[str, str] = {
    "run1 single-constraint": _WIRING,
    "run2 easy-multi": _WIRING,
    "run3 hard-multi": _WIRING,
    "run4 feedback": _WIRING,
    "run5 plan": _WIRING,
    "run6 cross-lineage": _WIRING,
    "run7 oracle": _WIRING,
    "run8 fix-budget": _WIRING,
    "run9 fb-actionability lo": _WIRING,
    "run10 fb-actionability hi": _WIRING,
    "D5-code MBPP+ (saturated)": _WIRING,
    # Not superseded by the wiring correction like the eleven above, but by the
    # blank-gene guard the run itself motivated. See EXPECTED_RUNS below.
    "C2 four-arm corrected harness": (
        "blank-gene guard refuses the empty reflection completion this cassette re-serves"
    ),
    # Same guard, a different run: replaying run11 raises on a blank `generate` gene. That is
    # informative rather than incidental -- Paper I quotes run11's worker swap, and the reader
    # is entitled to know its recording contains an accepted empty completion of exactly the
    # kind Paper III's defect 8 describes. The measured numbers stand; the replay does not.
    "run11 worker swap (nano)": (
        "blank-gene guard refuses an empty `generate` completion accepted at record time"
    ),
}

#: Released run -> why it is not registered as a replay row at all. Distinct from SUPERSEDED,
#: which holds runs that ARE registered (exact args frozen) and cannot replay. This was a prose
#: paragraph naming one BCB run; the release holds five, so the note covered a fifth of what a
#: reader would have taken it to cover. Machine-readable so `tests/test_verify_reproducibility.py`
#: can assert that every released run appears in exactly one of the two.
_BCB_NONDETERMINISM = (
    "BigCodeBench-Hard executes model code in a Docker sandbox. Model solutions may use "
    "unseeded randomness, so re-execution yields run-to-run variation that shifts the "
    "content-addressed prompt key -- an intrinsic property of executable-code benchmarks, not "
    "an oversight. Per-execution grading is deterministic and the headline reproduced "
    "identically across two independent paid runs (see Paper I, 'Reproducibility scope')."
)
UNREPLAYABLE: dict[str, str] = {
    "docs/validation-study-bcb.json": _BCB_NONDETERMINISM,
    "docs/validation-study-bcb-actuator.json": _BCB_NONDETERMINISM,
    "docs/validation-study-bcb-actuator-big.json": _BCB_NONDETERMINISM,
    "docs/validation-study-bcb-acthi.json": _BCB_NONDETERMINISM,
    "docs/validation-study-bcb-actlo.json": _BCB_NONDETERMINISM,
}

EXPECTED_RUNS = 25  # corrected harness: two C1 conditions, C2b, p0 L2/L3, the two OpenAI p0
# CALIBRATION passes (L2/L3, arms A -- these supply the p0 ladder Paper I's headline turns on
# and had been backing it from outside the guarantee), the three rows of the split-size sweep,
# the five bought rows of the budget sweep (D4), the cross-provider critic run (D3), and the
# eleven-run Groq second-worker line: three p0 calibration passes, five detector cells, and the
# verdict/prose splice G1. That line is registered for the same reason D3 is -- Paper II's
# `tab:secondworker` is built on it, so a run carrying an argument must not sit outside the
# replay guarantee.
# C2 and run11 are NOT among the 25. Auditing C2 found that two reflection completions came back
# empty and GEPA accepted one as the verify gene, so `genome_from_candidate` now refuses blank
# gene text -- and that guard makes C2's own cassette unreplayable, because replaying it
# re-serves the empty completion and the guard (correctly) raises. Its numbers stand as measured
# but are not verifiable under HEAD, which is exactly the state SUPERSEDED exists to record. C2b
# is the re-run under the guard and the raised reflection cap, and it does replay bit-for-bit.
# run11 fails the same guard on `generate`.

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
    # D3: the cross-provider critic. Registered here because all three papers quote it, and a
    # run whose numbers carry an argument while sitting outside the reproducibility guarantee is
    # the gap this script exists to close. It is also the only run whose critique surface is a
    # second vendor, so it is the one row that would notice a replay silently falling back to the
    # worker's provider -- the exact defect the `critic` cassette namespace was added to catch.
    (
        "D3 cross-provider critic",
        "docs/validation-study-xprovider.json",
        f"{CASSETTES}/study-xprovider.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "ABCD",
            "--seeds",
            "0,1,2,3,4,5,6,7",
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
            "--module-selector",
            "all",
            "--worker-model",
            "openai/gpt-4.1-mini",
            "--critique-model",
            "groq/llama-3.3-70b-versatile",
            "--reflection-model",
            "openai/gpt-5.6-terra",
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
    # The pre-registered split-size sweep: the alternative explanation for arm D's loss at
    # L1 was that eight training tasks is a small split. Metric calls per training task are
    # held at 3.5 so every row can fund an accept cycle -- a fixed budget could not, and the
    # harness now refuses that configuration at the dry-run step. n_train=8 is not a row
    # here because C2b already is it.
    (
        "split sweep n_train=4 (mmc=14)",
        "docs/validation-study-split4.json",
        f"{CASSETTES}/study-split4.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "BD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "4",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "14",
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
    (
        "split sweep n_train=16 (mmc=56)",
        "docs/validation-study-split16.json",
        f"{CASSETTES}/study-split16.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "BD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "16",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "56",
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
    (
        "split sweep n_train=24 (mmc=84)",
        "docs/validation-study-split24.json",
        f"{CASSETTES}/study-split24.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "BD",
            "--seeds",
            "0,1,2",
            "--n-train",
            "24",
            "--n-heldout",
            "10",
            "--max-metric-calls",
            "84",
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
    # The pre-registered budget sweep (D4). n_train is fixed at 8 throughout; what varies is
    # EFFECTIVE search depth = min(promotable cycles, --reflection-calls). The cap matters: at 6
    # it saturates at mmc=74, so the deepest cell raises it to 12 and is the only row that moves
    # two knobs. The cap is registered explicitly so the registry shows which limit bound.
    (
        "budget sweep mmc=20 (cap 6, depth 1.09)",
        "docs/validation-study-mmc20.json",
        f"{CASSETTES}/study-mmc20.json",
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
            "20",
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
    (
        "budget sweep mmc=40 (cap 6, depth 2.91)",
        "docs/validation-study-mmc40.json",
        f"{CASSETTES}/study-mmc40.json",
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
            "40",
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
    (
        "budget sweep mmc=56 (cap 6, depth 4.36)",
        "docs/validation-study-mmc56.json",
        f"{CASSETTES}/study-mmc56.json",
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
            "56",
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
    (
        "budget sweep mmc=80 (cap 6, depth 6.00)",
        "docs/validation-study-mmc80.json",
        f"{CASSETTES}/study-mmc80.json",
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
            "80",
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
    (
        "budget sweep mmc=160 (cap 12, depth 12.00)",
        "docs/validation-study-mmc160.json",
        f"{CASSETTES}/study-mmc160.json",
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
            "160",
            "--reflection-calls",
            "12",
            "--loop-max-retries",
            "1",
            "--module-selector",
            "all",
            "--reflection-model",
            "openai/gpt-5.6-luna",
        ],
    ),
    # The two OpenAI p0 CALIBRATION passes. Distinct from the `p0 sweep L2/L3` rows above:
    # those are the four-arm contrast runs, these are the single-arm passes that MEASURE p0
    # itself, and every p0 value in Paper I's headline axis is read off their cassettes. They
    # had been supplying that axis from outside the replay guarantee.
    (
        "p0 calibration L2 (arms A)",
        "docs/validation-study-p0cal-L2.json",
        f"{CASSETTES}/study-p0cal-L2.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "A",
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
        ],  # fmt: skip
    ),
    (
        "p0 calibration L3 (arms A)",
        "docs/validation-study-p0cal-L3.json",
        f"{CASSETTES}/study-p0cal-L3.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "A",
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
        ],  # fmt: skip
    ),
    # --- the Groq second-worker replication line (Paper II, sec:secondworker) -------------
    # Eleven runs on `groq/llama-3.3-70b-versatile`, every cell `--arms A` so no optimizer runs
    # anywhere. Registered for the reason given on D3 above: Paper II's `tab:secondworker` is
    # built on them, Paper I's closability table cites their p0 collapse, and Paper III's
    # defects 10-11 were found while buying them. A run that carries an argument does not get
    # to sit outside the guarantee.
    *[
        (
            label,
            f"docs/validation-study-groq-{slug}.json",
            f"{CASSETTES}/study-groq-{slug}.json",
            [
                "--task-set",
                "multi-hard",
                "--arms",
                "A",
                "--seeds",
                ",".join(str(i) for i in range(20)),
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
                "--module-selector",
                "all",
                "--worker-model",
                "groq/llama-3.3-70b-versatile",
                "--reflection-model",
                "groq/llama-3.3-70b-versatile",
                *extra,
            ],  # fmt: skip
        )
        for label, slug, extra in (
            # L1 is also Phase 1's INTRINSIC detector cell -- the worker judging its own prose
            # -- reused rather than re-bought, which is why one artifact carries two roles.
            ("groq p0 L1 / intrinsic", "p0cal-L1", ()),
            ("groq p0 L2", "p0cal-L2", ("--exclude-constraint", "exact_word_count")),
            (
                "groq p0 L3",
                "p0cal-L3",
                (
                    "--exclude-constraint",
                    "exact_word_count",
                    "--exclude-constraint",
                    "word_max_length",
                ),  # fmt: skip
            ),
            ("groq det 8b", "det-8b", ("--critique-model", "groq/llama-3.1-8b-instant")),
            ("groq det qwen", "det-qwen", ("--critique-model", "groq/qwen/qwen3.6-27b")),
            ("groq oracle mid", "det-oracle", ("--oracle-critique",)),
            ("groq oracle lo", "det-oracle-lo", ("--oracle-critique", "--feedback-level", "lo")),
            ("groq oracle hi", "det-oracle-hi", ("--oracle-critique", "--feedback-level", "hi")),
        )
    ],
    # The verdict/prose splice, G1 only: the diagonal cell, both roles on one model. G2-G4 were
    # never bought (see the Groq splice pre-registration's DEVIATION section). It is registered
    # because it is the run whose foreign worker enrolled as an eleventh sigma replicate, which
    # is Paper III's defect 11 -- the artifact that motivates a published guard has to be
    # verifiable, or the guard's evidence is a claim about a file nobody can re-derive.
    (
        "splice G1 (diagonal)",
        "docs/validation-study-splice-G1.json",
        f"{CASSETTES}/study-splice-G1.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "A",
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
            "--module-selector",
            "all",
            "--temperature",
            "0",
            "--worker-model",
            "groq/llama-3.1-8b-instant",
            "--critique-model",
            "groq/llama-3.1-8b-instant",
            "--verdict-model",
            "groq/llama-3.1-8b-instant",
            "--reflection-model",
            "groq/llama-3.1-8b-instant",
        ],  # fmt: skip
    ),
    # Registered and SUPERSEDED: Paper I quotes its worker swap, and replaying it raises on a
    # blank `generate` gene accepted at record time. Registering it is what makes that fact
    # visible; leaving it unregistered would have hidden a defect-8 instance inside a cited run.
    (
        "run11 worker swap (nano)",
        "docs/validation-study-multihard-nano.json",
        f"{CASSETTES}/study-multihard-nano.json",
        [
            "--task-set",
            "multi-hard",
            "--arms",
            "ABCD",
            "--seeds",
            "0,1,2",
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
            "--worker-model",
            "openai/gpt-4.1-nano",
            "--reflection-model",
            "openai/gpt-5.6-terra",
        ],  # fmt: skip
    ),
]


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
    pending = 0
    for label, committed, cassette, args in RUNS:
        if label in SUPERSEDED:
            skipped += 1
            print(f"[SKIP] {label:26} {SUPERSEDED[label]}")
            continue
        # A row can be registered BEFORE it is bought: the point of committing the exact args
        # ahead of the numbers is that the configuration cannot be edited to fit the result. An
        # absent artifact is therefore "not bought yet", not "does not reproduce", and conflating
        # the two would make the gate red for a pre-registration. It still does not count as
        # verified, so `complete` below stays false until the run actually lands.
        if not Path(committed).exists():
            pending += 1
            print(f"[PENDING] {label:26} registered, not yet bought ({committed} absent)")
            continue
        try:
            ok, detail = verify_one(committed, cassette, args)
        except Exception:  # noqa: BLE001 — a cassette miss OR a real code regression
            # Keep the traceback so a genuine bug is not misread as "just nondeterminism".
            ok, detail = False, traceback.format_exc().strip().splitlines()[-1]
        verified += 1
        all_ok &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {label:26} {detail}")
    pending_note = f"; {pending} registered but not yet bought" if pending else ""
    print(
        f"\n{skipped} run(s) skipped as superseded{pending_note}; "
        f"{verified} replayable run(s) checked."
    )
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
    elif all_ok and not complete:
        # Everything replayed exactly, but not everything registered has been bought. That is
        # INCOMPLETE, not FAILED, and saying "failed" here would read as a broken replay when the
        # truth is a pending purchase -- the same conflation the PENDING status above avoids.
        print(
            f"INCOMPLETE: all {verified} bought run(s) reproduce exactly, but "
            f"{EXPECTED_RUNS - verified} registered run(s) are not bought yet."
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
