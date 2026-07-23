# WhyStuck GEPA validation benchmark — design

**Date:** 2026-07-22
**Status:** design, implemented
**Scope:** one self-contained validation study that earns WhyStuck's central claim with empirical data, plus the reusable, deterministic harness that produces it.

> **Revision — 2026-07-22 (self-refine loop worker).** The worker described below was
> originally **single-shot** (one LLM call scores each task directly). A first offline
> validation showed that made the study's thesis untestable: IFBench's vendored rows are
> all **single-constraint**, and with a single-shot worker the `ensure_correct` verify gene
> is inert — arm D had no lever arm B lacks, so D and B tied at 0.69 (a null result driven
> by design, not by GEPA). The worker is now a real **generate→critique→refine loop**
> (`make_self_refine_run_task`): the `generate` gene drafts, the `ensure_correct` gene drives
> an *intrinsic* LLM critique (never the IFBench verifier — that is the scoring metric, so
> gating retries on it would be oracle contamination), and a refine step applies the critique.
> Evolving the verify gene now changes the critique and therefore the loop's output, which is
> the lever that separates loop-level (D) from prompt-only (B). Verified twice e2e against the
> real gepa library offline: controls A/C flat, D beats B by +1.0 on the synthetic landscape.
>
> **Superseded figures below.** Because the loop makes `1 + 2·loop_max_retries` LLM calls per
> task-eval, the cost model gained a `worker_calls_per_task` multiplier (`eval/cost.py`). The
> §4/§6 numbers assume the old single-shot worker (`31/31` split, `max_metric_calls`=150,
> ≈ $3.43); the **current default paid config** is `16/16`, `max_metric_calls`=64,
> `reflection_calls`=10, `loop_max_retries`=1 (3 calls/eval), ≈ **$4.54** over 2 seeds — under
> the $5 cap, which the guard enforces on the pre-flight estimate.
>
> **What the $5 guard does and does not guarantee** (corrected — the earlier draft overclaimed
> a "true bound"): call counts and *output* tokens are hard bounds — worker completions are
> capped at `max_tokens`=320 and reflection at 1500, wired through from `eval/cost.py`. *Input*
> tokens are a **conservative estimate**: the worker shape (1200 in, 320 out) covers the seed
> loop prompts plus a realistic GEPA-evolved instruction (~360 tokens), but GEPA can in
> principle propose a longer instruction (up to the reflection output cap) and sizes its own
> reflection prompt, neither capped here — so a pathological run could exceed the estimate. The
> guard checks the estimate pre-flight against `--max-usd`; it is not a runtime meter. The
> reflection-cap mechanism, arm definitions, controls, and honesty gates are unchanged.

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

`load_ifbench(split, n, seed) -> list[BenchmarkTask]` (payload carries `goal` = the IFBench prompt, and `instruction_id_list` = the constraints) plus two scorers over the verbatim-ported verifiers: `ifbench_pass(output, task) -> bool` (prompt-level: every constraint met) and `ifbench_score(output, task) -> float` (instruction-level: fraction met). The study feeds **`ifbench_score`** (fractional) to `make_run_task` as the `Score`, with **`ifbench_feedback`** (per-constraint prose) for reflection, and **reports that same fraction** on held-out — the objective GEPA climbs equals the metric we grade (no train/eval mismatch). `make_run_task` records the fractional score (`metadata['eval_score']`, which the adapter maximizes, falling back to `scalarize` for genome-blind command traces), the worker's output text, and the feedback on the trace, so GEPA's reflective dataset carries the real task/output/feedback rather than an internal status code. The authored suite uses the existing `keyword_score`.

### 4.3 `study.py` — arm orchestration

Defines the seed genome — a `generate` skill gene + an `ensure_correct` verifier-rubric gene (mirroring GEPA's 2-module IFBench agent) — the four arm configs, and `run_study(...)`.

**Fair-parity design:** all arms share the *identical* 2-gene seed content; arms differ only in each gene's `evolvable` flag (A evolves none, B evolves `generate` only, C and D evolve both). Since `render_text_genes` renders every text gene regardless of `evolvable`, the seed worker-prompt — and thus the seed held-out score — is identical across arms. That isolates "componentized loop-level optimization" (D) from "prompt-only" (B) without a "more prompt text" confound.

`run_study` returns per-arm `ArmResult{name, evolvable_genes, seed_heldout, opt_heldout, delta, variant_id}`. It drives the low-level `optimize_loop` directly (not the MCP wrappers `optimize_loop_via_llm`/`_via_commands`, which derive tasks from a dir/store rather than accepting explicit train/held-out splits; `_via_commands` also doesn't forward `_optimize`): A evaluates the seed with no optimization; B/D build `make_run_task(complete, ifbench_score, feedback=ifbench_feedback)`; C builds a genome-blind command run_task. The winning genome of each optimizing arm is scored on held-out with `ifbench_score` (graded). Cost/latency and the full objective vector are added by the step-4 runner / step-5 report, not here.

The GEPA `_optimize` seam and `Cassette`-wrapped models keep every arm runnable offline in tests. **Caveat:** arm C's *flatness* (genome-blind → GEPA finds no gradient → winner ≈ seed) is a real-GEPA behavior that an injected fake `_optimize` cannot reproduce, so the offline test covers C's *wiring* only; C's flatness is established by the live/E2E run and the existing BUG-2 landscape unit test.

### 4.4 entrypoint + outputs

- `eval/cost.py` — the budget rail (pure, unit-tested): a `MODEL_PRICES` table, `plan_study_calls(...)` (conservative per-call plan for a study config), `estimate_usd`, and `CostGuard.check` (raises `BudgetExceeded` over the ceiling). Kept out of the script so the pricing math is testable offline.
- `scripts/run_validation_study.py` — the one paid, live entrypoint, a thin CLI over `eval/cost.py`. **Defaults to a dry run** (prints the per-call breakdown + estimate, checks the ceiling, exits — no API key needed); spends only with explicit `--yes`, which additionally requires `OPENAI_API_KEY` and records every call to `--record`, and validates `--arms`/model names up front (a typo can't silently plan to $0). Both call counts are genuine upper bounds: worker via `max_metric_calls` (and the count includes `optimize_loop`'s post-search train re-scoring), reflection via `--reflection-calls`, which is wired to GEPA as a `MaxCandidateProposalsStopper` (one reflection LM call per proposal) so it is a *hard* cap, not an estimate — verified against the real gepa library offline. Real 2-seed config estimate: ≈ $3.43 (Terra, the default, under the $5 cap) / ≈ $5.38 (Sol, which the guard refuses at 2 seeds).
- `eval/report.py` — `render_report(runs, *, meta)` (pure, unit-tested): renders the Markdown report from a list of per-seed run dicts. Aggregates held-out **mean ± std across seeds**, and emits an honest verdict — **SUPPORTED** only if D beats B by more than the pooled seed noise, else **INCONCLUSIVE** / **NOT SUPPORTED**. With no runs it renders a *pending* report (full method, "Pending" results) so the doc exists before the paid run.
- `docs/validation-study.md` — the report, in nav and built by mkdocs. Committed now in its *pending* state; the step-7 live run overwrites it with real numbers + verdict. (Multi-seed is why the cost plan scales by `n_seeds`; the default 2-seed Terra run ≈ $3.43, and Sol × 2 exceeds the $5 cap — the guard forces the choice.)
- `tests/fixtures/cassettes/*.json` — committed recordings from the live run.

The internal design spec under `docs/superpowers/` is `exclude_docs`-ed from the mkdocs build (it has no nav entry by design), keeping `mkdocs build --strict` green.

## 5. Metrics

Report the full objective vector already emitted by `eval/objectives.py` (success/retry/diff efficiency), with IFBench held-out constraint-satisfaction as the primary. Every reported "optimized" variant carries the whole vector, never a single number (honesty gate). Also report per-arm cost (USD), latency, and wall-clock, and seed variance (≥ 2 seeds).

## 6. Models & budget

All OpenAI, via litellm + the user's `OPENAI_API_KEY` (WhyStuck stays provider-agnostic — only the model string changes):

- **worker / task LM:** `openai/gpt-4.1-mini` — GEPA's exact IFBench task LM. Small on purpose: leaves headroom for optimization to show a delta.
- **reflection LM:** GPT-5.6 family — **Terra ($2.50/$15)** default, **Sol ($5/$30)** for the final published run. Exact API id pinned at build time against OpenAI's live model list (`openai/gpt-5` may still alias).
- **budget:** hard cap **$5**, enforced on *both* axes. GEPA runs at `max_metric_calls = 150` per optimizing arm (bounds worker calls) and `--reflection-calls = 15` wired as a `MaxCandidateProposalsStopper` (bounds reflection calls — one per proposal; proven to hard-stop the real gepa search offline). Per optimizing arm the plan counts worker = `max_metric_calls + n_train + n_heldout` = 150 + 31 + 31 = 212 (search + winner train re-scoring + winner held-out) plus 31 for the seed held-out eval = 243; reflection = 15. Per seed the four arms total 579 worker calls (A 31, B 243, C 62, D 243) + 30 reflection (B, D). At worker c_w ≈ $0.00128/call and reflection c_r ≈ $0.0325/call (Terra), **2 seeds ≈ $3.43** (1158 worker + 60 reflection); Sol × 2 ≈ $5.38. The cost guard (§4.4), not this estimate, enforces the ceiling — it pre-counts planned calls and refuses over `--max-usd`; because the reflection cap is now a real stopper, the pre-count is a true upper bound rather than a hopeful one.

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
