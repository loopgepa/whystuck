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
.venv/bin/python scripts/analyze_evolved_gene.py   # the optimizer's own critique, as a detector
```

`analyze_evolved_gene.py` is the one to run if you read only one: it scores the rubric GEPA
actually evolved as a failure detector and finds it emits `PASS` zero times on every seed,
which is the mechanism behind the loop-level arm's loss.

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

Note `--reflection-model openai/gpt-5.6-luna` above. It is not the CLI default
(`gpt-5.6-terra`), and it is the reflection model the C2b re-run used — C2 and C2b are the
only two runs that did. Any run meant to compare against C2b must pin it, or the reflection
model becomes an uncontrolled difference.

### The p₀ sweep (§"The p₀ sweep: when loop-level optimization pays") — RUN

The experiment that turned the paper's null into a conditional result. `exact_word_count` is
satisfied on 0.16 of held-out final drafts and the generator forces it into *every* hard task,
so the tier's first-draft pass rate p₀ is pinned below 0.16. Excluding it raises p₀ while
keeping a brutal anchor in every task.

Measured p₀ and outcome (treatment reached `ensure_correct` on 3/3 seeds at every level):

| level | excluded | p₀ | Δ(D−B) | 95% CI | sign p |
| --- | --- | --- | --- | --- | --- |
| L1 | none (= C2b) | 0.067 | −0.042 | [−0.095, +0.050] | 1.00 |
| L2 | `exact_word_count` | 0.267 | **+0.060** | [+0.005, +0.105] | 0.25 |
| L3 | + `word_max_length` | 0.667 | −0.002 | [−0.045, +0.040] | 1.00 |

The sign flips between L1 and L2. Two caveats the paper states and this file repeats because
they are easy to drop: at n=3 with three same-signed deltas the exact interval **must** exclude
zero, so L2's interval is equivalent to sign-unanimity and the sign test is at its floor; and
**L3 is saturated** (baseline 0.942, both arms → ~0.97), which falsified our registered
monotonicity prediction for a reason that is a design error, not a rescue.

Measure p₀ from any cassette, free — the first critique event judged the first draft:

```bash
scratchpad/evalvenv/bin/python scripts/measure_p0.py
```

To reproduce the levels from scratch: one `--arms A` pass per level at **$0.09** creates the
cassette, then `--arms BD` at **$1.79** per level (both inside the $5 ceiling). Exact flags are
registered in `scripts/verify_reproducibility.py`; L1 needs no run because it *is* C2b.

`--exclude-constraint` refuses an id that names no live constraint, because excluding an
unknown id would run the un-ablated condition under an ablated label.

### Repeated samples and a pinned temperature (D7, partially built)

No interval in this study contains within-condition sampling variance: the worker draws one
sample per task at the provider's default temperature. Two of the three pieces now exist.

`--temperature` pins the worker's sampling temperature. It is passed **per call**, not baked
into the `litellm_complete` factory, so it enters the cassette key — a factory-level
temperature would not appear in the key and two runs at different temperatures would silently
serve each other's completions. Omitting the flag omits the parameter from the request body
entirely, so committed cassette keys are unchanged.

The worker also accepts a `sample=` salt whose only effect is on the cassette key. It is
required because the cassette dedupes identical calls in record mode: K identical sampling
calls would otherwise collapse to one entry and return byte-identical "samples", reading as
*zero* sampling variance — the most flattering artifact available. It is deliberately not sent
to the provider.

**Not built:** the repeat loop itself. Scoring threads through `run_task` and the GEPA adapter,
so the clean shape is a standalone offline re-scorer that rebuilds a released genome from its
`final_genes`, re-scores held-out K times with `sample=0..K-1` at a fixed temperature, and
emits per-(task, sample) scores. That needs no schema change and no re-optimization, since the
genomes are already released.

### The pre-registered split-size sweep (§"Is the loss a small-split artifact?")

Designed, guarded, priced, and pinned by `tests/test_split_size_sweep.py`; **not run** — the
credential available at the time authenticated against neither model. One command per row,
none of which needs the `$5` ceiling raised:

```bash
for cfg in "4 14" "16 56" "24 84"; do set -- $cfg
  scratchpad/evalvenv/bin/python scripts/run_validation_study.py \
    --task-set multi-hard --arms BD --seeds 0,1,2 \
    --n-train "$1" --n-heldout 10 --max-metric-calls "$2" --reflection-calls 6 \
    --loop-max-retries 1 --module-selector all \
    --reflection-model openai/gpt-5.6-luna \
    --record "tests/fixtures/cassettes/study-split$1.json" \
    --out "docs/validation-study-split$1.json" \
    --report "docs/validation-study-split$1.md" --yes
done
```

`n_train=8` is not in the loop because run C2b already is that row. Arms are `BD` only:
replaying C2b's cassette under `--arms BD` reproduces its B and D fields bit-for-bit, so arm A
(which re-scores a fixed genome) and arm C (flat by construction) add cost and nothing else.

Two things will bite anyone who varies `--n-train` without reading that section:

- **`--max-metric-calls` must scale with `--n-train`.** GEPA charges its seed evaluation
  `len(valset)` calls and the valset *is* the train split, so a fixed budget buys an arm that
  cannot complete one accept cycle, reports `genes_changed=[]`, and looks like a null. The
  harness now refuses this at the dry-run step; the sweep holds metric calls per training task
  at 3.5.
- **Keep `--n-train ≤ 31`.** The generated pool is `max(64, 2·max(n_train, n_heldout) + 2)`, so
  at 32 it leaves 64, and because pool size seeds the shuffle the held-out set silently moves.
  A test pins the cliff.

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
- **Two code tests skip on a loaded machine.** The code branch preflights every vendored
  reference solution through the sandbox before pricing a run, and that check is wall-clock
  bounded. On a busy host a correct reference (in practice `mbpp-271`) can exceed the cap,
  which `preflight_code_tasks` reports as a *host capacity* signal rather than bad data. The
  two tests that depend on it skip with that reason instead of failing, so a loaded machine
  does not manufacture a data-integrity error. The guard itself is unchanged and still
  refuses a live run, which is the case it exists for.
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

`verify_reproducibility.py` should print `ALL 5 RUNS REPRODUCE FROM CASSETTE`: the two C1
verifier-ablation conditions and the C2b four-arm re-run. Twelve further runs are reported as
skipped — eleven recorded under the pre-correction wiring (whose reflection prompts, and
therefore whose cassette keys, no longer exist) and C2 as above. The script exits non-zero if
it would otherwise claim success without having replayed anything.
