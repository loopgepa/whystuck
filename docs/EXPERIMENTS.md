# Experiments: every script, what it does, and how to reproduce it

Every number in the three papers comes from one of the scripts below. This page is the map from a
claim to the command that produced it.

**Two rules that shape everything here.**

1. **Exactly one script spends money.** `scripts/run_validation_study.py` is the only entrypoint
   that can reach a model API, and it will not spend a cent without `--yes`. Everything else in
   this table is `$0` and offline. If a script is not `run_validation_study.py` (or a driver that
   invokes it, marked below), it cannot bill you.
2. **Every paid call is recorded, so every paid run replays for free.** Calls go to a
   content-addressed cassette (`sha256` of name + arguments). `--replay` re-executes the whole
   pipeline against the cache with no API key, and a cassette miss raises rather than falling
   through to the network. That is why the 25 corrected-harness runs can be re-verified by anyone
   who clones the repository.

## Quick start: reproduce the papers without spending anything

Run every step from the repository root, with **both** virtual environments built as
[REPRODUCE.md](papers/REPRODUCE.md#setup) describes: the main `.venv` (no gepa) and the eval venv
`scratchpad/evalvenv` (with the `eval` extra). Several scripts below replay cassettes by spawning
`scratchpad/evalvenv/bin/python` through that relative path, so a different working directory or a
missing eval venv fails with a "broken environment" error rather than a wrong number.

```bash
# 1. every machine-checkable number in all three papers, against the released artifacts
.venv/bin/python scripts/verify_paper_claims.py            # 1685 checks, offline, $0, about a minute

# 2. the three-way split lost no quantity
.venv/bin/python scripts/verify_split_coverage.py          # 318 pre-split quantities accounted for, $0

# 3. every corrected-harness paid run replays bit-for-bit from its cassette
#    (runs in the eval venv; do not install the eval extra into .venv, which must stay gepa-free)
scratchpad/evalvenv/bin/python scripts/verify_reproducibility.py   # 25 verified, 13 skipped as superseded, $0

# 4. the figures, redrawn from the artifacts
#    (needs matplotlib, in the `figures` dependency group, not an extra:
#     uv pip install -p .venv --group figures; it also spawns the eval venv)
.venv/bin/python scripts/make_figures.py                   # writes docs/papers/shared/figures/*.pdf, $0

# 5. build the papers (needs pdflatex and bibtex; TeX Live 2023 or newer)
cd docs/papers/p1-conditional-lever && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

If step 1 or 2 fails, a paper disagrees with its data and the papers are wrong, not the scripts.

## The one paid entrypoint

### `run_validation_study.py`

The four-arm GEPA study. Everything paid in this project is one invocation of this.

```bash
# DRY RUN (default): prints the priced upper bound and exits without spending
scratchpad/evalvenv/bin/python scripts/run_validation_study.py --task-set multi-hard --arms ABCD --seeds 0,1,2 \
    --n-train 8 --n-heldout 10 --max-metric-calls 28 --out docs/validation-study-x.json

# SPEND: add --yes. Never do this without reading the dry run's estimate first.
scratchpad/evalvenv/bin/python scripts/run_validation_study.py ... --yes --record tests/fixtures/cassettes/x.json

# REPLAY a recorded run for free, no API key needed
scratchpad/evalvenv/bin/python scripts/run_validation_study.py ... --record tests/fixtures/cassettes/x.json --replay
```

| flag | what it controls |
|---|---|
| `--arms` | which of A/B/C/D to run. `A` scores a fixed genome that no optimizer can reach. Paper II's own cells ran `--arms A`; its curve reads arm A out of four-arm runs, where search ran on B and D only |
| `--seeds` | one genome pair per seed; the inferential unit |
| `--n-train` / `--n-heldout` | split sizes. Train and held-out never share a task id |
| `--max-metric-calls` | GEPA's budget. A configuration that cannot fund one accept cycle is **refused** at the dry-run step, because such a run bills in full and reports a flat Δ with `genes_changed=[]` |
| `--reflection-calls` | hard cap on proposals per arm. Effective search depth is `min(budget, cap)`, and forgetting the second one invalidated a sweep |
| `--max-usd` | pre-flight ceiling, default `$5`. Five runs raised it explicitly (up to `$11.76`) |
| `--oracle-critique` | replaces the in-loop gate with the true metric. **Deliberate oracle contamination**, admissible only as a ceiling experiment |
| `--feedback-stats` | gives the critique `text_stats(draft)`, a non-oracle measurement |
| `--feedback-level` | `lo` / `mid` / `hi` repair text. The Paper II actionability lever |
| `--loop-max-retries` | refine budget *R*. Cost per eval is `1 + 2R` calls |
| `--sample k` | replicate index. **Enters the cassette key**, so record-mode dedup cannot serve K replicates one cached draw |
| `--task-set multi-hard-manifest` | reads tasks from a frozen manifest instead of the generator; needs `--task-manifest` and `--manifest-level`. Used by the target-p₀ study on the third worker |
| `--task-manifest PATH` | the manifest JSON (`whystuck.eval.task_manifest`): the tasks, and per level, seed and split which of them are scored. Re-verified against the generator on load, so an edited task is refused |
| `--manifest-level NAME` | which manifest level one run scores (`calib`, or a level such as `low` / `mid` / `high`) |
| `--empty-attempts N` | calls per request while the model keeps returning empty content. Default 2, what every run before the target-p₀ study used; that study ran at 6. Each resample is counted into the run's metadata (`empty_resamples`); a `--replay` records 0 there, because the cassette serves the final non-empty completion, so read the count from the released JSON |

The `--sample` note is not a detail. Without the salt, repeated scorings of one genome are
byte-identical calls, the cache returns the same draw K times, and the measured variance is
**exactly zero** — a confident, wrong noise floor.

## Paid drivers (they shell out to the entrypoint)

### `repeat_scores.py` — the replicate loop

Re-scores already-released genomes K times to put a sampling interval on Δ. No re-optimization, so
this is the ordinary scoring path on a fixed genome.

```bash
scratchpad/evalvenv/bin/python scripts/repeat_scores.py            # p0 family: dry run, prints the priced bound
scratchpad/evalvenv/bin/python scripts/repeat_scores.py --curve    # detector-curve family: dry run
scratchpad/evalvenv/bin/python scripts/repeat_scores.py --yes      # spend
scratchpad/evalvenv/bin/python scripts/repeat_scores.py --verify   # replay the 135 p0-family replicates, $0
scratchpad/evalvenv/bin/python scripts/repeat_scores.py --curve --verify   # replay the 90 curve replicates, $0
```

Two families, both fully bought and released:

| family | runs | priced bound | feeds |
|---|---|---|---|
| p0 levels | 135 (3 levels × {B, D, seed} × 3 seeds × 5 reps) | — | Paper I `tab:repeat`, Paper III `tab:sigmagenome` |
| detector curve | 90 (6 rows × 3 seeds × 5 reps) | `$2.98` | Paper II `tab:curvesigma`, `tab:curvepairs` |

All 225 replay offline at `$0`: `--verify` checks the 135 of the p0 family and
`--curve --verify` the 90 of the curve family, so both commands are needed to cover all of them. The curve family exists because every ratio in Paper II had been
divided by a σ measured on a *different* genome family, and three of six had less margin against
the 2σ line than the known spread in σ.

### `positive_control.py`

Builds a landscape where evolving the verify gene is the **only** path to a passing answer, then
runs the real four-arm study through real GEPA on it. No network, no spend.

```bash
scratchpad/evalvenv/bin/python scripts/positive_control.py      # expects A/B/C = 0.000, D = 1.000
```

It rules out a dead measurement path and nothing more: run against the pre-fix source tree it
prints the same four scores and passes, so it would not have caught the defects Paper III
retracts. A null from an instrument never shown to register a signal is still not evidence.

## Offline analyzers ($0, no network, no key)

Each reads released JSON and cassettes and derives one family of claims.

| script | derives | used by |
|---|---|---|
| `analyze_deltas.py` | per-seed Δ, exact-resample intervals, sign tests | I, II |
| `analyze_delta_decomposition.py` | Δ split into the shared `generate` term and the loop term | I `tab:decomp` |
| `analyze_critique_detector.py` | the critique scored as a binary detector (precision, F₁, Youden's *J*, MCC) | II `tab:curve` |
| `analyze_evolved_gene.py` | whether an **evolved** critique gained discrimination or only moved its threshold; the `PASS`-rate screen | I §analysis, I §splitresult |
| `analyze_residual_by_constraint.py` | per-constraint satisfaction of the loop's final draft | II `tab:residual` |
| `analyze_measurement_noise.py` | the cross-run noise floor σ(Δ) from runs that re-score one fixed genome | I §threats, III §sigma |
| `analyze_search_depth.py` | the search depth each run **actually** reached, by counting GEPA iterations in the cassette | I `tab:budgetresult` |
| `analyze_c1.py` | task-level paired analysis of the hand-written verify gene | I appendix |
| `measure_p0.py` | p₀ per exclusion level, from any recorded cassette | I §p0 |
| `p0_task_stats.py` | per-task first-draft pass share (`p_first`) and mean final score from arm-A calibration cassettes of a manifest level; strict replay | target-p₀ study |
| `build_p0_levels.py` | builds the study manifest's `low` / `mid` / `high` levels from those stats with `task_manifest.select_levels`, under the pre-registered matching rule | target-p₀ study |
| `analyze_p0target.py` | the pre-registered analysis of the three four-arm runs, written to `docs/p0target/analysis.json` | target-p₀ study, I D9 |
| `analyze_qwen_l1l2.py` | the pre-registered analysis of Paper I's L1/L2 design on the third worker, written to `docs/qwen-l1l2/analysis.json` | third-worker L1/L2 test, I |
| `replay_qwen_l1l2.py` | strict end-to-end replay of that test's two runs against the release, and its measured spend floor, written to `docs/qwen-l1l2/replay.json` | third-worker L1/L2 test |

`analyze_search_depth.py` deserves a note: the pre-registration *computed* depth from a formula
that charged every proposal a full valset re-evaluation, which only an accepted proposal pays. The
formula understated depth, three sweep cells silently landed on one effective depth, and the fix was
to stop computing depth and measure it.

## Verifiers and gates

| script | asserts |
|---|---|
| `verify_paper_claims.py` | 1685 checks: every machine-checkable number in the three papers against the artifacts, plus per-paper spine numbers, cross-paper agreement, and each paper's corrections count |
| `verify_split_coverage.py` | all 318 quantities in the pre-split draft are carried by some paper, superseded by a verified replacement, or explicitly dropped with a reason |
| `verify_reproducibility.py` | every corrected-harness run replays bit-for-bit; superseded runs are **skipped and reported**, never silently "verified" |
| `verify_stdio.py` | the MCP console script speaks stdio (product, not paper) |
| `check_public_claims.sh` | the README's test-count claim is not an overstatement |
| `check_versions.sh` | `pyproject` and the plugin manifest agree on version |
| `dev_check.sh` | the full local gate: ruff, format, mypy, pytest |

### Why `verify_paper_claims.py` reads all three papers at once

It concatenates the papers (shared `\input`s expanded) via `scripts/paper_sources.py`. "Does the
prose agree with the artifact" is a question about the series, not about which file a sentence
landed in — and it means the ~155 pinned-phrase checks written against the pre-split draft still
work: a sentence that survived into *some* paper is found, and one the split dropped yields `NaN`
and fails. Per-paper questions get their own check families, because concatenation cannot answer
them.

## One-time vendoring tools (dev only)

`vendor_bigcodebench.py` and `vendor_mbpp_plus.py` download a benchmark once and commit a pinned,
sandbox-verified subset. The study must run with no network, so benchmark data is vendored rather
than fetched. You should not need to re-run these; if you do, the committed JSON changes and every
code-domain cassette key changes with it.

## Released artifacts

| path | contents |
|---|---|
| `docs/validation-study-*.json` | 43 study runs, per-arm and **per-task** held-out vectors, evolved gene text, per-gene accepted-change counts, and the pre-flight cost estimate |
| `docs/repeat/*.json` | 225 replicate scorings (135 p₀ + 90 curve) |
| `tests/fixtures/cassettes/**` | 284 cassettes; this is what makes replay free |
| `docs/papers/PRE-SPLIT-SOURCE.tex` | the 46-page pre-split draft, frozen at `fe628ed`, **never built**. Provenance for the split proof |
| `docs/actual-spend.json` | measured spend per run and per provider, from an instrumented replay. See below |
| `docs/p0-manifests/` | the target-p₀ study's task manifests and per-task calibration stats (`pilot-1000*`, `full-1001-*`), the calibration runs (`calib/`), and the study manifest `study.json`. Outside the 43-run count |
| `docs/p0target/` | the study's three four-arm runs (`validation-study-p0target-{low,mid,high}`), `analysis.json` and `REPORT.md`. Third worker `groq/qwen/qwen3.8-27b`, reflection `groq/openai/gpt-oss-120b`; pre-registration `docs/superpowers/specs/2026-10-25-target-p0-preregistration.md` |
| `docs/qwen-l1l2/` | the third-worker L1/L2 test: two four-arm runs (`validation-study-qwen-{L1,L2}`), `analysis.json`, `replay.json` and `REPORT.md`. Same worker and reflector as the target-p₀ study; pre-registration `docs/superpowers/specs/2026-10-26-qwen-l1-l2-replication-preregistration.md`. Outside the 43-run count |

38 of the 43 runs also carry a generated `docs/validation-study-*.md` report. The five without one
are the bought rows of the pre-registered budget sweep (D4), where the suffix is the
`--max-metric-calls` value: `validation-study-mmc20.json`, `validation-study-mmc40.json`,
`validation-study-mmc56.json`, `validation-study-mmc80.json` and
`validation-study-mmc160.json`. They carry
committed cassettes, they replay, and Paper I's `tab:budgetresult` is built from them; only the
report writer was never run over them. Named here because until 2026-08-13 no prose document in the
repository mentioned them at all, so five released runs feeding a published table had no page
describing them. `tests/test_paper_docs_consistency.py::test_every_released_run_is_named_in_prose`
now refuses that state.

Each run JSON records a pre-flight **bound** on spend, not an invoice. Bounds are labelled as
bounds throughout the papers; an earlier draft reported one as a measured spend, and that is in the
corrections register.

### What it actually cost, and why the bounds are ~10× high

```bash
scratchpad/evalvenv/bin/python scripts/measure_actual_spend.py   # offline, $0; writes docs/actual-spend.json
```

Run it from the repository root in the eval venv: it replays through gepa and counts tokens with
`tiktoken`, which the eval extra brings in (through litellm) and the main `.venv` lacks. It took
16 s on 2026-10-26 and about 90 s on another machine the same day, and its output was byte-identical
to the committed `docs/actual-spend.json` both times. It covers the 43-run release and the replicate
loop; the target-p₀ program's spend is not in it (see `docs/p0target/REPORT.md`).

Prompts are stored as hashes, so a cassette cannot price its own input tokens — but a replay
reconstructs every prompt byte-for-byte, which is what makes replay match at all. The script
replays each run with all four call surfaces instrumented, counts **distinct** (surface, prompt)
pairs, and prices them from `MODEL_PRICES`. Distinct, not fired: during the original recording a
repeated prompt was served from the cassette and never billed, so D3 bills 347 calls where a
replay fires 1124.

| | runs | USD |
|---|---|---|
| measured, both sides | 25 | **$4.97** |
| estimated, outputs measured and inputs modelled from the measured per-surface ratio | 18 | **$4.73** |
| the replicate loop, same basis | 225 scorings | **$1.39** |
| **floor for the whole study** (the 43-run release; the target-p₀ program is not in it) | | **$11.08** |
| sum of the pre-flight bounds, all 43 runs | | $114.61 |

Of that, **$0.73** is the borrowed Groq credential, against $400 authorised.

A **floor**, not an invoice, and the three reasons are worth knowing before quoting it: a call
that failed and was retried was billed and left no cassette entry; `cl100k_base` is neither
Llama's tokenizer nor gpt-4.1's, and on D3 a chars/4 cross-check spanned $0.074–$0.084; and the 18
unreplayable runs have a modelled input side, which is why they are never summed into the measured
figure.

The measurement also tests something no guard tested before: **the pre-flight bound is supposed to
be a ceiling, and nothing had ever compared one to an outcome.** It held on all 43 runs, with the
tightest margin at 3.1× and the loosest at 14.9×. `tests/test_actual_spend.py` asserts it keeps
holding, and asserts the method still reproduces the D3 figure that was measured by hand two days
before the script existed — 347 calls, 95 937 input tokens, 22 193 output, to the token.

## Reading a `genes_changed` field before you read a score

The single most useful field in any run JSON:

```json
"genes_changed": ["generate", "ensure_correct"]
```

Per component, whether the optimizer's output differs from the seed. Paper III exists because the
first fifteen contrast runs did not print it: replayed at the pre-fix commit, ten of the twelve
replayable runs had a critique gene byte-identical to the seed in every winning genome (the other
two moved it on one seed each, and only five of the twelve could have shown a move;
`scripts/replay_superseded_genes.py`), ten runs reported held-out
scores identical on every seed, and the consistency read as a tight null when it was a genome
compared against itself. Check this field before
interpreting any Δ. Its companion screen is as cheap — a `PASS` rate of zero means the search went
to the degenerate corner.
