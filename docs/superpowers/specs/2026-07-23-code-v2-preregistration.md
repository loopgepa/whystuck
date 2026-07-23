# Pre-registration — v2 Cross-Domain, Code (MBPP+)

**Date:** 2026-07-23
**Status:** pre-registered, BUILD IN PROGRESS, no spend yet. Frozen once the paid run starts; no post-hoc changes to arms, benchmark, split protocol, seeds, budget, or the reported metric.

## Motivation

v1 (10 runs) found loop-level GEPA does not beat prompt-only on IFEval-style
instruction-following, and traced *why*: (a) the intrinsic critique is a near-random
detector, and (b) even a perfect detector with maximally-actionable feedback can't fix
the residual (exact word-counting). Our mechanism predicts loop-level should help only
where **verification is reliable (executable, not LLM self-grading)** AND **the fix is
reachable**. Code satisfies both: unit tests are ground-truth verification, and "edit
code until tests pass" is a real, actionable loop — the tool/executor regime where
self-correction is known to work (Reflexion; Self-Refine). v2-code is the fair test of
the boundary.

## Benchmark choice (frozen): MBPP+ (EvalPlus)

**MBPP+** \[Liu et al. 2023, arXiv 2305.01210]: entry/intermediate Python function tasks,
each with a base test suite and an augmented "plus" edge-case suite. Chosen over the
contamination-free alternative (LiveCodeBench) for three decisive reasons:

1. **Reproducibility.** Our project guarantees bit-for-bit replay. MBPP+ tasks are fast,
   pure functions → execution is deterministic under a generous timeout. LiveCodeBench's
   competitive problems are timing-sensitive (tight limits) → pass/fail can flip across
   replays → would break reproducibility. **Determinism wins.**
2. **Safety.** Pure functions, no external libraries, no file/network I/O → a simple,
   sound subprocess sandbox suffices.
3. **A natural anti-oracle split** (below), which also restores headroom and softens
   contamination.

**Contamination caveat (stated honestly):** MBPP is older than the worker's training
cutoff, so some memorization is possible. Mitigations: (i) we score on the *plus* tests,
whose edge cases catch shallow/memorized solutions; (ii) headroom is verified empirically
before committing (calibration below). **LiveCodeBench \[Jain et al. 2024] is the
contamination-clean robustness follow-up**, accepting its reproducibility cost, if MBPP+
saturates or the result is close.

## The anti-oracle protocol (the honesty crux)

- **In-loop verification = the BASE tests (visible).** The critique/refine step may run
  the base tests and see which failed + the traceback. This is reliable, executable
  verification — exactly what v1's intrinsic critique lacked. (HF `mbppplus` schema: the
  base suite is the `test_list` field.)
- **Scoring = the PLUS tests (hidden).** The reported held-out score runs the augmented
  plus suite (the `test` field's generated `check()`), which the loop never sees.

**Precise claim (plus ⊇ base):** EvalPlus's plus suite is the base tests *plus* extra
generated edge-case assertions, not a wholly disjoint set. So the loop cannot see or
optimize against the **plus-only edge cases** — those are the hidden signal, and passing
the visible base tests does NOT guarantee passing them (that gap is the headroom EvalPlus
was built to expose). The programmatic scorer is never consulted in-loop, so this is not
oracle contamination; but we do not overclaim full base/plus disjointness. This mirrors
real coding agents (optimize against the given tests, held out on stricter hidden ones) and
is the setting where evolving *how the critique reads a failing test* (arm D) could finally
beat prompt-only (arm B).

## Arms (unchanged from v1)

- **A** seed baseline (no optimization) — the readout for the loop's raw fix ability.
- **B** prompt-only GEPA (evolve `generate`).
- **C** genome-blind command control (flatness check).
- **D** loop-level GEPA (evolve `generate` + `ensure_correct`); the verify gene shapes
  how the critique consumes execution feedback.

Quantity of interest: Δ(D−B) on the hidden-test held-out score, with an exact-resample CI.

## Worker loop (reuses `make_self_refine_run_task`)

generate → run base tests → critique (reads failing base test + traceback, gated by the
`ensure_correct` gene) → refine → repeat up to `loop_max_retries`. Final draft scored on
the plus tests. The base-test execution is the in-loop signal (like the oracle gate was
run 7's, but here it's a *realistic, non-oracle* tool: visible tests ≠ scoring tests).

## Sandbox (frozen)

Subprocess running `sys.executable`, with: `RLIMIT_CPU` + `RLIMIT_AS` (memory) limits, a
wall-clock timeout (generous, e.g. 10 s — MBPP+ tasks run in ms, so the limit only guards
runaways and does not affect pass/fail on correct code), a fresh temp cwd, and stdout/stderr
captured. **Honest limitation:** subprocess isolation is not container-grade; it is
acceptable here because the executed code is `gpt-4.1-mini` output (non-adversarial) and
MBPP+ tasks need no I/O/network. Prefer a container/VM if available.

**Determinism contract:** scoring is a pure function of the candidate code for these
deterministic, non-timing-sensitive tasks; the timeout is set well above the tasks' true
runtime so it never arbitrates a correct solution. This is what lets `--replay` reproduce
the held-out scores bit-for-bit even though execution is not cassette-recorded (the LLM
calls are; execution is deterministic-local, like arm C's command runner).

## Headroom / calibration (pre-committed)

Expected seed (arm A) hidden-test score band: **0.45–0.80** (mid; gpt-4.1-mini ≈0.70 on
MBPP+ full). Flags: ceiling if any seed ≥0.9, floor if ≤0.1 → uninformative (as in runs 1–2).
If a pre-spend pilot (a few tasks) shows saturation, curate to the harder MBPP+ subset
before committing — the difficulty calibration is fixed *before* the scored run, not after.

## Metrics & analysis (frozen)

- **Primary:** Δ(D−B) = mean_seed(D.opt_heldout − B.opt_heldout) on the plus (hidden) tests,
  with an exact-resample 95% CI (as in v1's `analyze_deltas.py`).
- **Secondary:** seed vs optimized per arm; C flat control; per-task failure taxonomy
  (which test category resists) as the code analog of v1's per-constraint decomposition.
- **Verdict rule:** SUPPORTED if Δ(D−B) CI excludes 0 and is positive; NOT SUPPORTED if the
  CI brackets 0; report as-is, no p-hacking. Report even a surprising positive plainly.

## Budget

`gpt-4.1-mini` worker, `gpt-5.6-terra` reflection. Code outputs are longer than IFEval →
higher cost/call; size n_heldout (~12–15), n_train (~8), seeds (3), mmc, and
loop_max_retries so the conservative estimate stays well under the $5 cap. **Dry-run
estimate shown BEFORE any spend; no `--yes` until the offline e2e is green and the estimate
is approved.**

## Build order (TDD; de-risk before data + before spend)

1. **`code_exec.py`** — the sandboxed executor (highest risk); TDD against hand-written
   pass / fail / timeout / crash / import-error cases. **← this commit.**
2. `code_bench.py` — vendored MBPP+ loader (base/plus split), `code_score` (plus/hidden),
   `code_feedback` (failing base test + traceback), satisfiability check (the reference
   solution passes both suites).
3. Wire into `run_study` (reuse arms/cassette); offline e2e through real GEPA with a faked
   worker; a pilot to confirm headroom.
4. Pre-flight dry-run estimate → show → (approval) → paid run → record cassette → verify
   reproducibility → report → review → merge. Same discipline as v1.

## Implementation status (2026-07-23) — built + verified offline, NO spend yet

Steps 1–3 are done and merged; step 4 awaits approval. Key implementation decisions,
each pinned by tests:

- **Data.** 64 tasks vendored to `src/whystuck/eval/data/mbpp_plus.json` (ascending
  `task_id`, lean+fast, every reference verified to pass both suites at vendoring time).
  Network-free at run time; ships in the wheel.
- **Sandbox / determinism.** The MBPP+ plus harness imports numpy; `code_exec` pins
  `OPENBLAS/OMP/MKL_NUM_THREADS=1` (keeps numpy's virtual arena under `RLIMIT_AS` *and*
  makes reductions deterministic) and raises the default `RLIMIT_AS` to 2 GB (the memory
  bomb guard still holds). numpy added to the `eval` extra + dev group.
- **In-loop signal.** An additive `exec_feedback(draft, task)` seam in the self-refine
  worker injects the VISIBLE base-test results into the *non-oracle* critique (the
  Reflexion regime v1 lacked); off by default so prior cassettes replay byte-identically.
- **Anti-oracle enforcement.** `code_score` = hidden plus; `code_feedback`/`exec_feedback`
  = visible base only; the plus-test text reaches no prompt and no reflection signal. The
  CLI refuses `--oracle-critique --task-set code` (would gate on the hidden scorer).
- **Pre-spend correctness review (done, before any spend).** Two adversarial reviewers.
  The code-reviewer cleared all four e2e priorities (scoring, anti-oracle, determinism,
  budget) under execution. The silent-failure hunter found and we fixed a real class of
  bug: infrastructure faults (a `sandbox_error` / `no_assertions` from a broken run host
  or corrupt data) must NOT be folded into a genuine-looking `0.0` — that would silently
  zero a paid run and read as a null result. `ExecResult` now carries a structured
  `verdict`; `code_score`/`code_feedback` **raise** on an infra verdict (a wrong answer /
  timeout / early-exit still scores `0.0`), and `preflight_code_tasks` runs every vendored
  reference through the sandbox at the **dry-run step** so a broken host/data crashes
  before a cent is spent.
- **Recommended config + estimate.** 6 train / 10 held-out, `max_metric_calls=40`,
  8 proposals, 1 loop retry, 2 seeds → conservative dry-run ≈ **$4.34** (under the $5 cap;
  the code worker is priced at the larger `CODE_WORKER_TOKENS = (2200, 512)`). Fewer
  seeds/tasks than v1's 16/16 to stay under budget with the larger code token shape; a
  pre-spend pilot will confirm headroom (seed band 0.45–0.80) before the scored run.
- **Verification.** 458 tests pass (incl. an end-to-end run through `run_study` with real
  sandboxed execution); ruff/format/mypy/mkdocs-strict green.
