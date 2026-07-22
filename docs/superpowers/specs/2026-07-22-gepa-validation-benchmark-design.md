# WhyStuck GEPA validation benchmark — design

**Date:** 2026-07-22
**Status:** design, pending implementation-plan
**Scope:** one self-contained validation study that earns WhyStuck's central claim with empirical data, plus the reusable, deterministic harness that produces it.

## 1. Goal

Produce the first empirical evidence that WhyStuck's optimize path actually improves loops — specifically the headline research claim:

> **loop-level GEPA (optimizing the whole loop genome) beats prompt-only GEPA (optimizing a single instruction) under a matched budget.**

Everything currently green in the repo is plumbing proven with fakes; the optimize path has never run against a real LLM. This study closes both gaps at once: it runs the marquee feature live for the first time, and it reports a numbers table that earns the word "optimized" (the honesty gate in `docs/v0.2-eval-benchmark.md`).

Secondary outcome, free from the same harness: a live end-to-end integration test of `optimize_loop_via_llm` (cassette-backed), closing the "only fakes" coverage gap.

## 2. The claim, made falsifiable

Four arms, identical worker / reflection model / budget / task set, differing only in what GEPA is allowed to evolve:

| Arm | What GEPA evolves | Path | LLM cost |
|-----|-------------------|------|----------|
| **A** | nothing (seed genome, no optimization) | direct eval | worker only |
| **B** | **one** instruction gene (prompt-only GEPA) | `optimize_loop_via_llm`, single-gene genome | worker + reflection |
| **C** | whole genome, **command** worker (genome-blind) | `optimize_loop_via_commands` | **$0** (no LLM) |
| **D** | whole **multi-gene** genome, LLM worker (loop-level GEPA) | `optimize_loop_via_llm`, full genome | worker + reflection |

Expected ordering on held-out score: A < B < D, with C ≈ A (flat — the control that empirically demonstrates BUG 2's premise: a genome-blind worker can't move text genes).

**Thesis verdict** = D's held-out score − B's held-out score, reported as mean ± std across seeds. The claim holds iff that gap exceeds seed noise. A null or negative result is a legitimate, publishable outcome — the study is designed to be able to refute the thesis, not just confirm it.

## 3. Dataset

**IFBench** (`allenai/IFBench_test`, Hugging Face) — instruction-following with 58 programmatic constraint verifiers (Pyatkin et al., arXiv 2507.02833, NeurIPS 2025 D&B). GEPA's paper (Agrawal et al., arXiv 2507.19457) is a *separate* paper that evaluates on IFBench; that's precisely why B-vs-D on this dataset is a faithful test against the "prompt-only GEPA" baseline. Chosen because:

- It makes B-vs-D an apples-to-apples test against the exact "prompt-only GEPA" baseline the thesis names, on a dataset the method's authors published on.
- Verification is code, not an LLM judge → deterministic scoring, no judge cost, no judge variance.
- Short completions → cheap; the score is fraction-of-constraints-satisfied ∈ [0, 1].

Only the **parameter-free, nltk-free, package-free** subset of IFBench's 58
constraints is ported (11 constraints — `format:parentheses`, `format:quotes`,
`count:punctuation`, `words:palindrome`, `format:no_whitespace`,
`custom:multiples`, `words:vowel`, `words:consonants`, `format:newline`,
`format:quote_unquote`, `format:sub-bullets`), so scoring is fully deterministic
and offline with no per-row kwargs. The vendored slice
(`src/whystuck/eval/data/ifbench_supported.jsonl`) is exactly the rows those
constraints fully cover: **62 rows**, split deterministically per seed into
**31 train / 31 held-out** (the reported number comes from held-out). This is
smaller than GEPA's 150/300/294 protocol; the study claims a *relative* result
(loop-level > prompt-only) under matched budget, not a leaderboard number, and
the report states the split size honestly. Constraints needing `nltk`, an
`emoji` package, or per-row kwargs were excluded by design.

A **tier-0 authored micro-task suite** (~8–12 tasks, `{goal, expect}` with
substring/regex verifiers, scored by the existing `keyword_score`) exists first,
purely to build and prove the harness at $0 before any IFBench spend.

**IFBench license (verified, split):** the ported verifier *logic* is **Apache-2.0**
(codebase license); the vendored data *rows* are **ODC-BY-1.0** (data license).
Both permit this redistribution with attribution
(`src/whystuck/eval/data/ATTRIBUTION.md`).

## 4. Architecture

New code under `src/whystuck/eval/` (opt-in `whystuck[eval]`), three units with clean seams:

### 4.1 `cassette.py` — deterministic record/replay

Both LLM surfaces are pure `input → str` callables, so both are wrappable:

- worker: `complete(prompt) -> str` (already `litellm_complete(model)`)
- gepa reflection: `reflection_lm(messages, **kwargs) -> str` (gepa accepts a callable — confirmed in gepa FAQ)

`Cassette.wrap(fn, name=...)` wraps either, keyed by a stable sha256 of `(name, args, kwargs)`. Mode `record` calls through to the live model and stores `{name, output}` per key in a JSON file (the input is folded into the hash key, not stored verbatim); mode `replay` returns the stored output and raises `CassetteMiss` on a miss. One live run records; CI and every re-run replay from committed JSON — no HTTP-level VCR needed, no tokens burned in CI. IFBench verifiers are already deterministic. *(Implemented in `eval/cassette.py`, PR #14.)*

### 4.2 `ifbench.py` — loader + programmatic scorer

`load_ifbench(split, n, seed) -> list[BenchmarkTask]` (payload carries `goal` = the IFBench prompt, and `instruction_id_list` = the constraints) plus two scorers over the verbatim-ported verifiers: `ifbench_pass(output, task) -> bool` (prompt-level: every constraint met) and `ifbench_score(output, task) -> float` (instruction-level: fraction met). The study feeds **`ifbench_pass`** to `make_run_task` as the `Score` (a clean binary GEPA signal that plugs into the existing genome→trace→scalarize path unchanged) and **reports both** on held-out. The authored suite uses the existing `keyword_score`.

### 4.3 `study.py` — arm orchestration

Defines the seed genome — a `generate` skill gene + an `ensure_correct` verifier-rubric gene (mirroring GEPA's 2-module IFBench agent) — the four arm configs, and `run_study(...)`.

**Fair-parity design:** all arms share the *identical* 2-gene seed content; arms differ only in each gene's `evolvable` flag (A evolves none, B evolves `generate` only, C and D evolve both). Since `render_text_genes` renders every text gene regardless of `evolvable`, the seed worker-prompt — and thus the seed held-out score — is identical across arms. That isolates "componentized loop-level optimization" (D) from "prompt-only" (B) without a "more prompt text" confound.

`run_study` returns per-arm `ArmResult{name, evolvable_genes, seed_heldout, opt_heldout, delta, variant_id}`. It drives the low-level `optimize_loop` directly (not the MCP wrappers `optimize_loop_via_llm`/`_via_commands`, which derive tasks from a dir/store rather than accepting explicit train/held-out splits; `_via_commands` also doesn't forward `_optimize`): A evaluates the seed with no optimization; B/D build `make_run_task(complete, ifbench_pass)`; C builds a genome-blind command run_task. The winning genome of each optimizing arm is scored on held-out with `ifbench_score` (graded). Cost/latency and the full objective vector are added by the step-4 runner / step-5 report, not here.

The GEPA `_optimize` seam and `Cassette`-wrapped models keep every arm runnable offline in tests. **Caveat:** arm C's *flatness* (genome-blind → GEPA finds no gradient → winner ≈ seed) is a real-GEPA behavior that an injected fake `_optimize` cannot reproduce, so the offline test covers C's *wiring* only; C's flatness is established by the live/E2E run and the existing BUG-2 landscape unit test.

### 4.4 entrypoint + outputs

- `eval/cost.py` — the budget rail (pure, unit-tested): a `MODEL_PRICES` table, `plan_study_calls(...)` (conservative per-call plan for a study config), `estimate_usd`, and `CostGuard.check` (raises `BudgetExceeded` over the ceiling). Kept out of the script so the pricing math is testable offline.
- `scripts/run_validation_study.py` — the one paid, live entrypoint, a thin CLI over `eval/cost.py`. **Defaults to a dry run** (prints the per-call breakdown + estimate, checks the ceiling, exits — no API key needed); spends only with explicit `--yes`, which additionally requires `OPENAI_API_KEY` and records every call to `--record`, and validates `--arms`/model names up front (a typo can't silently plan to $0). Worker-call counting is a genuine upper bound (it includes `optimize_loop`'s post-search train re-scoring); `reflection_calls` is a caller estimate. Real config estimate: ≈ $1.72 (Terra) / $2.69 (Sol), both under the $5 cap.
- `docs/validation-study.md` — the report: the 4-arm numbers table (held-out score mean ± std, full objective vector, cost, latency, wall-clock), method, and verdict. Regenerated from the study's JSON output.
- `tests/fixtures/cassettes/*.json` — committed recordings from the live run.

## 5. Metrics

Report the full objective vector already emitted by `eval/objectives.py` (success/retry/diff efficiency), with IFBench held-out constraint-satisfaction as the primary. Every reported "optimized" variant carries the whole vector, never a single number (honesty gate). Also report per-arm cost (USD), latency, and wall-clock, and seed variance (≥ 2 seeds).

## 6. Models & budget

All OpenAI, via litellm + the user's `OPENAI_API_KEY` (WhyStuck stays provider-agnostic — only the model string changes):

- **worker / task LM:** `openai/gpt-4.1-mini` — GEPA's exact IFBench task LM. Small on purpose: leaves headroom for optimization to show a delta.
- **reflection LM:** GPT-5.6 family — **Terra ($2.50/$15)** default, **Sol ($5/$30)** for the final published run. Exact API id pinned at build time against OpenAI's live model list (`openai/gpt-5` may still alias).
- **budget:** hard cap **$5**. GEPA runs at `max_metric_calls = 150` (its default) per optimizing arm; costing uses a conservative R ≈ 300 worker calls/arm (the 150 optimization rollouts + held-out + baseline evals) and K ≈ 15 reflection calls/arm. With worker c_w ≈ $0.0013/call and reflection c_r ≈ $0.033/call (Terra): per optimizing arm ≈ 300·0.0013 + 15·0.033 ≈ $0.88 ⇒ B + D ≈ $1.76; extra baselines/held-out via OpenAI Batch API (−50%) ≈ $0.13; 2 seeds ≈ $0.9 ⇒ **≈ $2.8 total** (≈ $3.9 with Sol). The cost guard (§4.4), not this estimate, enforces the ceiling — it pre-counts planned calls and refuses over `--max-usd`.

## 7. Determinism & CI

CI never calls a live model. The E2E integration test runs the whole study through `Cassette` (replay mode against committed cassettes and asserts: (a) `optimize_loop_via_llm` runs end-to-end and its winner beats the seed, (b) arm D's landscape is non-flat while arm C's is flat, (c) replay is byte-deterministic. Same-input replay → 100% hit.

## 8. Build order (each increment TDD, own PR)

1. `cassette.py` record/replay + tests (fakes only).
2. `ifbench.py` loader + `ifbench_score` + tests (fixture constraints, no network).
3. `study.py` seed genome + arm configs + `run_study`, tested offline via injected `_optimize` + a fake `complete` over a deterministic single-constraint IFBench fixture (`format:no_whitespace`).
4. `scripts/run_validation_study.py` with the cost guard + dry-run estimator; unit-test the estimator and the ceiling refusal.
5. Report generator (`docs/validation-study.md` from JSON) + a smoke test.
6. E2E replay test (§7) — the coverage-gap closer.
7. **The one live run** (manual, gated): records cassettes, fills the report with real numbers. Not part of CI.

Steps 1–6 are $0 and land before any spend. Step 7 is the single paid action, run only on explicit go.

## 9. Non-goals & risks

**Non-goals:** SWE-bench (blows $5, needs Docker, doesn't isolate the thesis); a real-LLM CI; a SOTA claim — we claim the *relative* result (loop-level > prompt-only) under matched budget, not a leaderboard number.

**Risks:** (a) exact GPT-5.6 id — pin at build. (b) gepa `reflection_lm` callable signature (`messages` vs `prompt`) — verify against the installed gepa before wiring. (c) IFBench licensing — verify before vendoring; authored-suite fallback. (d) token overrun — the cost guard pre-counts planned calls and refuses over `--max-usd`. (e) null result — treated as a valid finding, reported honestly.
