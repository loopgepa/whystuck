# Reproducing the papers

This directory holds the **three papers** of the series ([README.md](README.md) says which one
answers which question). Everything below runs offline and costs **$0** unless a step says
otherwise.

Paper III's central methodological claim is that a componentized-optimization result should
report, per component, whether the optimizer's output actually differs from the seed. That
claim is only credible if our own numbers can be re-derived rather than taken on trust, so
each check below re-derives a different part of the series from the released
artifacts, and each one fails loudly on drift.

For the full map from a claim to the script that produced it, see
[../EXPERIMENTS.md](../EXPERIMENTS.md).

## Setup

Two virtual environments, deliberately:

```bash
# 1. Main venv — everything except live runs. gepa is NOT installed here on purpose,
#    so a test asserting an actionable "gepa is missing" error stays meaningful.
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 2. Eval venv — only needed to replay cassettes or make a live run (needs gepa + litellm).
python -m venv scratchpad/evalvenv && scratchpad/evalvenv/bin/pip install -e '.[eval]'
```

## The checks

### 1. Every number in all three papers matches the artifacts ($0, ~30s)

```bash
.venv/bin/python scripts/verify_paper_claims.py
```

Re-derives every machine-checkable figure the three papers print — the detector table, the
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
.venv/bin/python scripts/analyze_delta_decomposition.py  # how much of Δ is actually the treatment
```

`analyze_evolved_gene.py` is the one to run if you read only one: it scores the rubric GEPA
actually evolved as a failure detector and finds it emits `PASS` zero times on every seed,
which is the mechanism behind the loop-level arm's loss.

`analyze_delta_decomposition.py` is the one to run second, because it qualifies the headline.
Both arms evolve `generate`, so Δ(D−B) is a whole-system contrast; the script splits it into a
draft term (the shared gene) and a loop term (the treatment) and prints four blocks:

| block | what it answers |
|---|---|
| the split | 60% of L1's −0.042 is the shared gene; 100% of L2's +0.060 is the treatment |
| strata on the seed genome's first draft | the L1 loss is **not** on already-correct drafts (+0.002) |
| refine productivity | refine is net-positive at every level, so no destructive-repair story |
| answer length by arm | the do-nothing floor is not being exploited (D is not shorter) |

The strata use the **seed** genome's draft, not each arm's own, because an arm's draft is
downstream of the treatment: stratifying on it would condition on a collider.

`analyze_deltas.py` marks any run whose seeds share a task pool, because on those the
interval and the sign test are anti-conservative and must not be read as replication.

### 5. The three-way split lost no claim ($0, ~5s)

```bash
.venv/bin/python scripts/verify_split_coverage.py
```

The papers were carved out of one 46-page draft. A split is a mass edit of prose, and its failure
mode is silent: a paragraph belonging to no paper's argument gets dropped and nothing errors. This
compares the frozen pre-split source against the three papers and requires every quantity to be
carried by at least one of them, superseded by a replacement it verifies is present, or explicitly
dropped with a reason. All 322 are accounted for and none is dropped.

Also runs as tests (`tests/test_split_coverage.py`), including one that re-reads the pre-split draft
from git to prove the baseline was not edited to make the check pass.

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

Designed, guarded, priced, and pinned by `tests/test_split_size_sweep.py`, and since **run**: all
three rows are released as `docs/validation-study-split{4,16,24}.json`, registered in
`verify_reproducibility.py`, and they replay. This section read "**not run** — the credential
available at the time authenticated against neither model" until 2026-08-13, which was true when
written and stayed on the page after the credential worked. The commands are kept because they are
the record of what was run, and none of them needs the `$5` ceiling raised:

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
| `docs/papers/p1-conditional-lever/main.tex` | Paper I — the conditional lever |
| `docs/papers/p2-repair-not-detection/main.tex` | Paper II — repair, not detection |
| `docs/papers/p3-silent-no-op/main.tex` | Paper III — the silent no-op |
| `docs/papers/shared/` | preamble, harness section, bibliography, figures, CC BY licence |
| `docs/papers/PRE-SPLIT-SOURCE.tex` | the pre-split draft, frozen at `fe628ed`, never built |
| `docs/validation-study-*.json` | per-run results, including per-task held-out vectors |
| `docs/validation-study-*.md` | the generated report, for 38 of the 43 runs (see below) |
| `tests/fixtures/cassettes/` | recorded LLM calls, content-addressed, prompts not stored |
| `docs/genes/` | hand-written genes used in the C1 ablation, with rationale |
| `scripts/` | the study runner and every offline analysis |
| `src/whystuck/eval/` | harness: adapter, study, cassette, cost guard, benchmarks |

### The five runs with no generated report

`docs/validation-study-mmc{20,40,56,80,160}.json` are the five bought rows of the pre-registered
budget sweep (D4). They carry committed JSON and cassettes, they are registered in
`verify_reproducibility.py`, they replay, and Paper I's budget-sweep table is built from them — but
the report writer was not run for them, so they have no `.md` companion.

That is worth stating rather than leaving as an arithmetic puzzle. Counting `.json` against `.md`
gives 43 against 39 and suggests four are missing; the real figure is five, because
`docs/validation-study-summary.md` is a hand-written running log with no `.json` sibling and it
masks one of them. This row of the table above claimed a report for every run until 2026-08-13.
`tests/test_paper_docs_consistency.py::test_every_released_run_is_named_in_prose` now requires each
released run to be named somewhere in the documentation, which is what was actually missing: these
five were released, cited by a paper table, and named in no prose at all.

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

`verify_reproducibility.py` should print `ALL 25 RUNS REPRODUCE FROM CASSETTE`. Those are the
corrected-harness runs: the two C1 verifier-ablation conditions, the C2b four-arm re-run, the p0
levels and the two OpenAI calibration passes, the split-size sweep, the five bought rows of the
budget sweep, the cross-provider critic run, and the eleven-run Groq second-worker line.

A further **13** runs are reported as skipped as superseded, each carrying its own reason in the
registry: eleven recorded under the pre-correction wiring (whose reflection prompts, and therefore
whose cassette keys, no longer exist) and two the blank-gene guard now refuses, because replay
re-serves the empty completion that motivated the guard. The remaining **5** of the 43 released runs
are listed as unreplayable — the BigCodeBench ones, whose scoring executes model code in a Docker
sandbox and is not bit-reproducible offline. 25 + 13 + 5 = 43.

The script exits non-zero if it would otherwise claim success without having replayed anything.

This paragraph said `ALL 5 RUNS` and "Twelve further runs" until 2026-08-13. Eleven runs had been
replaying for some time without being registered, so the counts moved to 25/13 in
`docs/EXPERIMENTS.md` and `docs/papers/README.md` — both of which a test pins — and this file, which
no test read, kept the old numbers. It is now in the same guard
(`tests/test_paper_docs_consistency.py::test_advertised_replay_counts_equal_the_registry`).
