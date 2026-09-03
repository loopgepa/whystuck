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

```bash
# 1. every machine-checkable number in all three papers, against the released artifacts
python scripts/verify_paper_claims.py            # 1036 checks, offline, $0

# 2. the three-way split lost no quantity
python scripts/verify_split_coverage.py          # 318 pre-split quantities accounted for, $0

# 3. every corrected-harness paid run replays bit-for-bit from its cassette
#    (needs the eval extra: pip install -e '.[eval]')
python scripts/verify_reproducibility.py         # 25 verified, 13 skipped as superseded, $0

# 4. the figures, redrawn from the artifacts
#    (needs matplotlib, in the `figures` dependency group, not an extra:
#     uv pip install -p .venv --group figures)
python scripts/make_figures.py                   # writes docs/papers/shared/figures/*.pdf, $0

# 5. build the papers (needs pdflatex and bibtex; TeX Live 2023 or newer)
cd docs/papers/p1-conditional-lever && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

If step 1 or 2 fails, a paper disagrees with its data and the papers are wrong, not the scripts.

## The one paid entrypoint

### `run_validation_study.py`

The four-arm GEPA study. Everything paid in this project is one invocation of this.

```bash
# DRY RUN (default): prints the priced upper bound and exits without spending
python scripts/run_validation_study.py --task-set multi-hard --arms ABCD --seeds 0,1,2 \
    --n-train 8 --n-heldout 10 --max-metric-calls 28 --out docs/validation-study-x.json

# SPEND: add --yes. Never do this without reading the dry run's estimate first.
python scripts/run_validation_study.py ... --yes --record tests/fixtures/cassettes/x.json

# REPLAY a recorded run for free, no API key needed
python scripts/run_validation_study.py ... --replay tests/fixtures/cassettes/x.json
```

| flag | what it controls |
|---|---|
| `--arms` | which of A/B/C/D to run. `A` scores a fixed genome and runs **no optimizer** — this is how every Paper II measurement was taken |
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

The `--sample` note is not a detail. Without the salt, repeated scorings of one genome are
byte-identical calls, the cache returns the same draw K times, and the measured variance is
**exactly zero** — a confident, wrong noise floor.

## Paid drivers (they shell out to the entrypoint)

### `repeat_scores.py` — the replicate loop

Re-scores already-released genomes K times to put a sampling interval on Δ. No re-optimization, so
this is the ordinary scoring path on a fixed genome.

```bash
python scripts/repeat_scores.py            # p0 family: dry run, prints the priced bound
python scripts/repeat_scores.py --curve    # detector-curve family: dry run
python scripts/repeat_scores.py --yes      # spend
python scripts/repeat_scores.py --verify   # replay every bought replicate, $0
```

Two families, both fully bought and released:

| family | runs | priced bound | feeds |
|---|---|---|---|
| p0 levels | 135 (3 levels × {B, D, seed} × 3 seeds × 5 reps) | — | Paper I `tab:repeat`, Paper III `tab:sigmagenome` |
| detector curve | 90 (6 rows × 3 seeds × 5 reps) | `$2.98` | Paper II `tab:curvesigma`, `tab:curvepairs` |

All 225 replay offline at `$0`. The curve family exists because every ratio in Paper II had been
divided by a σ measured on a *different* genome family, and three of six had less margin against
the 2σ line than the known spread in σ.

### `positive_control.py`

Builds a landscape where evolving the verify gene is the **only** path to a passing answer, then
runs the real four-arm study through real GEPA on it. No network, no spend.

```bash
python scripts/positive_control.py      # expects A/B/C = 0.000, D = 1.000
```

This is the only artifact in the project that would have failed loudly under the original,
defective wiring. A null from an instrument never shown to register a signal is not evidence.

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

`analyze_search_depth.py` deserves a note: the pre-registration *computed* depth from a formula
that charged every proposal a full valset re-evaluation, which only an accepted proposal pays. The
formula understated depth, three sweep cells silently landed on one effective depth, and the fix was
to stop computing depth and measure it.

## Verifiers and gates

| script | asserts |
|---|---|
| `verify_paper_claims.py` | 1036 checks: every machine-checkable number in the three papers against the artifacts, plus per-paper spine numbers, cross-paper agreement, and each paper's corrections count |
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
| `tests/fixtures/cassettes/**` | 267 cassettes; this is what makes replay free |
| `docs/papers/PRE-SPLIT-SOURCE.tex` | the 46-page pre-split draft, frozen at `fe628ed`, **never built**. Provenance for the split proof |
| `docs/actual-spend.json` | measured spend per run and per provider, from an instrumented replay. See below |

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
python scripts/measure_actual_spend.py   # ~40 min, offline, $0; writes docs/actual-spend.json
```

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
| **floor for the whole study** | | **$11.08** |
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
first fifteen contrast runs did not print it: the critique gene was byte-identical to the seed in
15/15 winning genomes, ten runs reported a held-out difference of exactly zero, and the consistency
read as a tight null when it was a genome compared against itself. Check this field before
interpreting any Δ. Its companion screen is as cheap — a `PASS` rate of zero means the search went
to the degenerate corner.
