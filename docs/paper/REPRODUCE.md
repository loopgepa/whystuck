# Reproducing the paper

This directory is the supporting artifact for *Your Loop Optimizer Is Not Optimizing Your
Loop*. Everything below runs offline and costs **$0** unless a step says otherwise.

The paper's central methodological claim is that a componentized-optimization result should
report, per component, whether the optimizer's output actually differs from the seed. That
claim is only credible if our own numbers can be re-derived rather than taken on trust, so
each of the four checks below re-derives a different part of the paper from the released
artifacts, and each one fails loudly on drift.

## Setup

Two virtual environments, deliberately:

```bash
# 1. Main venv — everything except live runs. gepa is NOT installed here on purpose,
#    so a test asserting an actionable "gepa is missing" error stays meaningful.
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. Eval venv — only needed to replay cassettes or make a live run (needs gepa + litellm).
python -m venv scratchpad/evalvenv && scratchpad/evalvenv/bin/pip install -e '.[eval]'
```

## The four checks

### 1. Every number in the paper matches the artifacts ($0, ~30s)

```bash
.venv/bin/python scripts/verify_paper_claims.py
```

Re-derives every machine-checkable figure the paper prints — the detector table, the
residual decomposition, both actionability effects with intervals and sign tests, the
metric floors, the oracle taxonomy, the C1 ablation, the C2 re-run, and the count of runs
that overrode the spend ceiling — directly from `docs/*.json`, the vendored task data, and
the committed cassettes. It exits non-zero on any mismatch and prints, at the end, the
claims it *cannot* check, so the gap is visible rather than implied.

Also runs as a parametrized test (`tests/test_verify_paper_claims.py`), one case per claim,
plus a guard that the checklist has not silently emptied.

### 2. The committed runs replay bit-for-bit ($0)

```bash
scratchpad/evalvenv/bin/python scripts/verify_reproducibility.py
```

Replays each run recorded under the corrected harness from its cassette, with no network and
no key, and compares every per-arm score against the committed result. Runs recorded under
superseded wiring are reported as skipped rather than silently counted as verified — the
script refuses to print a success banner if nothing was actually replayed.

Replay is strict: a cassette miss raises, and since a miss inside GEPA's reflection channel
used to be swallowed (Evidence 7), the cassette now counts misses and raises at context exit
too.

### 3. The instrument responds to a real effect ($0, needs the eval venv)

```bash
scratchpad/evalvenv/bin/python scripts/positive_control.py
```

Every headline in the paper is a null, so this is the control that makes those nulls
readable. It builds a landscape where evolving the verify gene is the *only* path to a
passing answer, runs the real four-arm study through real GEPA, and asserts that the
loop-level arm beats the prompt-only arm. Expected: A/B/C at `0.000`, D at `1.000`,
Δ(D−B) = `+1.000`.

### 4. The offline analyses ($0)

```bash
.venv/bin/python scripts/analyze_deltas.py              # per-run Δ(D−B), exact-resample CIs, sign tests
.venv/bin/python scripts/analyze_critique_detector.py   # the critique as a failure detector (F1)
.venv/bin/python scripts/analyze_residual_by_constraint.py
.venv/bin/python scripts/analyze_c1.py docs/validation-study-c1-base.json \
                                       docs/validation-study-c1-hand.json
```

`analyze_deltas.py` marks any run whose seeds share a task pool, because on those the
interval and the sign test are anti-conservative and must not be read as replication.

## Making a live run (costs money)

```bash
export OPENAI_API_KEY=...
scratchpad/evalvenv/bin/python scripts/run_validation_study.py \
  --task-set multi-hard --arms ABCD --seeds 0,1,2 \
  --n-train 8 --n-heldout 10 --max-metric-calls 28 --reflection-calls 6 \
  --loop-max-retries 1 --module-selector all \
  --reflection-model openai/gpt-5.6-luna \
  --max-usd 3.00 \
  --record tests/fixtures/cassettes/study-new.json \
  --out docs/validation-study-new.json --report docs/validation-study-new.md
```

Without `--yes` this prints a priced estimate and spends nothing. The estimate is a genuine
**ceiling** on output tokens (they are hard-capped via `max_tokens`) and a conservative
estimate on input tokens; measured realization has been 20–25% of the bound. Add `--yes` to
spend. The cassette persists after every call, so an interrupted run loses nothing and a
re-invocation resumes from what was already paid for.

## Layout

| path | what it holds |
| --- | --- |
| `docs/paper/main.tex` | the paper |
| `docs/paper/references.bib` | bibliography; every entry verified against its source |
| `docs/validation-study-*.json` | per-run results, including per-task held-out vectors |
| `docs/validation-study-*.md` | the generated report for each run |
| `tests/fixtures/cassettes/` | recorded LLM calls, content-addressed, prompts not stored |
| `docs/genes/` | hand-written genes used in the C1 ablation, with rationale |
| `scripts/` | the study runner and every offline analysis |
| `src/whystuck/eval/` | harness: adapter, study, cassette, cost guard, benchmarks |

## Gates

All four must pass before any commit:

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```

## Known limits of this artifact

Stated here rather than discovered:

- **Cassettes do not store prompts.** They are content-addressed by
  `sha256(name, args, kwargs)`, so any change to a prompt changes every key. This is why the
  eleven pre-correction runs cannot replay under the corrected harness.
- **Executable-code runs do not replay byte-for-byte.** Model solutions may use unseeded
  randomness, so re-executing them can yield varying test output. Scores and cassettes are
  released; full-trace replay is not claimed for the BigCodeBench runs.
- **The provenance fields postdate most runs.** `genes_changed`, `train_score` and
  `final_genes` are empty in the older result files, so the reporting minimum the paper
  proposes cannot be checked against those runs — only against ones made after it.
- **One run is deliberately unreplayable.** `validation-study-c2.json` is the contaminated
  first attempt at the four-arm re-run: two of its reflection completions came back empty and
  one was accepted as the verify gene. The guard added in response
  (`genome_from_candidate` refuses blank gene text) means replaying that cassette now raises,
  which is correct — the recording contains a candidate the harness must refuse. It is kept
  as the evidence for that finding, marked superseded, and `validation-study-c2b.json` is the
  clean re-run that the paper's numbers come from.

## Which runs verify

`verify_reproducibility.py` should print `ALL 3 RUNS REPRODUCE FROM CASSETTE`: the two C1
verifier-ablation conditions and the C2b four-arm re-run. Twelve further runs are reported as
skipped — eleven recorded under the pre-correction wiring (whose reflection prompts, and
therefore whose cassette keys, no longer exist) and C2 as above. The script exits non-zero if
it would otherwise claim success without having replayed anything.
