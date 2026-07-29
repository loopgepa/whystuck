#!/usr/bin/env python
"""Check every machine-checkable number in the paper against the released artifacts.

Zero cost, offline. The paper quotes dozens of numbers derived from the committed results
JSONs and cassettes; each was correct when written, and each is one careless edit away from
no longer being. This script re-derives them from the artifacts and fails loudly on any
drift, so "the paper agrees with its data" is a test rather than a memory.

It exists because a fact-check pass found two real errors that had survived several
readings: the paper said three runs overrode the \\$5 default when four did, and it said
the reproducibility script registers eleven runs after two more had been added. Both are
the kind of quiet arithmetic drift that a reviewer finds and an author does not.

A later pass added the Evidence-8 checks, which exist for a sharper reason: the ad-hoc
integrity scan that should have caught an empty reflection completion was itself vacuous
(it serialised each cassette entry to JSON before measuring length, so it compared the size
of a dictionary and could never see an empty output). Those numbers are pinned here so the
check has to keep working rather than merely having worked once.

A third pass added the `tex:` checks, for the same reason one level up. Every other check
compares an artifact against a number typed into *this file*, so it verifies the data and
not the paper; the two can disagree. They did: the closability table claimed the corrected
re-run changed `ensure_correct` on 2/3 seeds while the data and the body text both say 3/3
(the 2/3 is arm B's `generate`, transposed). The `tex:` checks scrape main.tex itself, so
the claimed side is the paper's own sentence. A pattern that stops matching yields NaN and
fails, because a reworded sentence must break the check loudly rather than quietly stop
being checked.

Scope: numbers recomputable from `docs/*.json`, the vendored task data, and the committed
cassettes, plus the headline numbers as they are written in `docs/paper/main.tex`. Claims
that are not machine-checkable (unquantified prose, cited literature, one-off replays
against superseded commits) are out of scope and listed at the bottom of the output so the
gap is visible rather than implied.

    python scripts/verify_paper_claims.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean, pstdev

DOCS = "docs"
PAPER = f"{DOCS}/paper/main.tex"

# The pre-registered split-size sweep. Kept here as the single source of truth so the
# paper table, the guard tests and these checks cannot drift apart.
SWEEP = ((4, 14), (8, 28), (16, 56), (24, 84))
EPOCHS = 3.5  # metric calls per training task, held constant across the sweep
POOL_CLIFF = 32  # first n_train whose pool leaves 64, which would move the held-out set


def _meta_of(name: str) -> dict:
    """A released run's `meta` block. Mirrors `_runs`, using plain `open` so it does not depend
    on the lazy `pathlib` imports the analyzer shims use."""
    with open(f"{DOCS}/{name}", encoding="utf-8") as fh:
        return json.load(fh)["meta"]


def _runs(name: str) -> list[dict]:
    with open(f"{DOCS}/{name}", encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _tex() -> str:
    with open(PAPER, encoding="utf-8") as fh:
        return fh.read()


#: Per-constraint final-draft satisfaction on held-out, run 3 (intrinsic, R=1), from
#: tab:residual. The p0 estimate is built from these, so if that table changes this must too.
_RESIDUAL_RATES = {
    "exact_word_count": 0.16,
    "word_max_length": 0.85,
    "forbidden_word": 0.92,
    "num_bullets": 1.00,
    "end_with": 0.95,
    "include_keyword": 1.00,
    "all_lowercase": 0.95,
    "no_comma": 0.95,
}
#: (excluded constraints, E[p0], max p0) per level of the pre-registered p0 sweep.
_P0_LEVELS = (
    (frozenset(), 0.127, 0.16),
    (frozenset({"exact_word_count"}), 0.725, 0.85),
    (frozenset({"exact_word_count", "word_max_length"}), 0.832, 0.92),
)


#: Per p0 level: (label, cassette, exclusions, pinned confusion matrix tp/fp/fn/tn).
#: The pinned matrices are the measured ones; they are the fallback when the eval extra is
#: absent, and the live replay must agree with them (tests/ exercise that path).
_P0_DETECTOR = (
    ("L1", "study-c2b.json", frozenset(), (1, 6, 1, 22)),
    ("L2", "study-p0cal-L2.json", frozenset({"exact_word_count"}), (5, 15, 3, 7)),
    (
        "L3",
        "study-p0cal-L3.json",
        frozenset({"exact_word_count", "word_max_length"}),
        (12, 6, 8, 4),
    ),
)


#: Results JSON per p0 level. L1 is the four-arm re-run itself.
_P0_RUN_FILES = (
    "validation-study-c2b.json",
    "validation-study-p0-L2.json",
    "validation-study-p0-L3.json",
)


def _p0_deltas(level: int) -> list[float]:
    """Per-seed delta(D-B) at one p0 level, via the same estimator the paper reports."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_deltas import delta_arm

    return list(delta_arm(f"{DOCS}/{_P0_RUN_FILES[level]}"))


def _p0_ci(level: int) -> tuple[float, float]:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_deltas import exact_resample_ci

    _, lo, hi = exact_resample_ci(_p0_deltas(level))
    return lo, hi


def _confusion(level: int) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) for the seed critique as a detector at one p0 level."""
    _, cassette, exclude, pinned = _P0_DETECTOR[level]
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_critique_detector import collect_events, detector_stats

        stats = detector_stats(
            collect_events(
                f"tests/fixtures/cassettes/{cassette}",
                [0, 1, 2],
                loop_kw={},
                n_heldout=10,
                exclude=exclude,
            )
        )
        return int(stats["tp"]), int(stats["fp"]), int(stats["fn"]), int(stats["tn"])
    except Exception:
        return pinned


def _detector_metrics(level: int) -> dict[str, float]:
    """p0, F1 and the two base-rate-INDEPENDENT statistics, from the confusion matrix.

    Youden's J and Matthews' correlation are the point of this table: F1 rises with the base
    rate whether or not the detector improves, so a rising F1 on an easier benchmark says
    nothing about discrimination. J and MCC are zero at chance for any base rate.
    """
    import math

    tp, fp, fn, tn = _confusion(level)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else math.nan
    rec = tp / (tp + fn) if tp + fn else math.nan
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "p0": (tp + fn) / n,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else math.nan,
        "sens": sens,
        "spec": spec,
        "j": sens + spec - 1,
        "mcc": (tp * tn - fp * fn) / den if den else math.nan,
    }


def _measured_p0_l1() -> tuple[int, int]:
    """(already-correct first drafts, first drafts) at L1, replayed from the C2b cassette.

    Needs the eval extra (the replay drives the real loop), so it degrades to the pinned
    2/30 when gepa/litellm are absent rather than failing the whole check run under the
    no-eval venv. The degradation is loud in the sense that matters: the pinned pair is the
    same pair the measurement produced, and `tests/` exercises the live path.
    """
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_critique_detector import collect_events

        events = collect_events(
            "tests/fixtures/cassettes/study-c2b.json", [0, 1, 2], loop_kw={}, n_heldout=10
        )
        return sum(1 for _, actual in events if actual), len(events)
    except Exception:
        return 2, 30


def _expected_p0(exclude: frozenset[str]) -> tuple[float, float]:
    """(E[p0], max p0) for one exclusion level, from the generator and the residual table.

    E[p0] multiplies the per-constraint rates over each generated task and averages over the
    three seeds, so it assumes independence and uses final-draft rates -- an ordering and
    magnitude estimate, which is all the pre-registration claims. `max p0` is the ceiling set
    by the hardest constraint present in *every* task at that level, which is what makes the
    hard tier's p0 low: `exact_word_count` is in all of them and is satisfied 0.16 of the time.
    """
    from statistics import mean

    from whystuck.eval.multi_constraint import generate_multi_tasks

    products, common = [], None
    for seed in (0, 1, 2):
        tasks = generate_multi_tasks(64, seed=seed, k_min=4, k_max=5, hard=True, exclude=exclude)
        for t in tasks:
            ids = t.payload["instruction_id_list"]
            p = 1.0
            for i in ids:
                p *= _RESIDUAL_RATES[i]
            products.append(p)
            common = set(ids) if common is None else (common & set(ids))
    ceiling = min((_RESIDUAL_RATES[c] for c in (common or ())), default=1.0)
    return mean(products), ceiling


def _type_overlap_counts(n_train: int, n_heldout: int = 10) -> list[int]:
    """Per seed: how many training tasks have a constraint-type set also seen in held-out.

    The sweep's attribution fails if a larger split simply contains more near-copies of
    held-out tasks. Recomputed from the generator here rather than trusted from the table.
    """
    import ast

    from whystuck.eval.multi_constraint import load_multi_constraint

    pool = max(64, 2 * max(n_train, n_heldout) + 2)

    def payload(task: object) -> dict:
        raw = task.payload  # type: ignore[attr-defined]
        return raw if isinstance(raw, dict) else ast.literal_eval(raw)

    counts = []
    for seed in (0, 1, 2):
        held = [
            frozenset(payload(t)["instruction_id_list"])
            for t in load_multi_constraint("heldout", seed=seed, n=n_heldout, pool=64, hard=True)
        ]
        train = load_multi_constraint("train", seed=seed, n=n_train, pool=pool, hard=True)
        hit = sum(1 for t in train if frozenset(payload(t)["instruction_id_list"]) in held)
        counts.append(hit)
    return counts


def _table_row(tex: str, header_marker: str, row_label: str) -> list[list[float]]:
    """All numbers, per cell, of one LaTeX table row, located by its header.

    Per cell rather than one-number-per-cell because a column may legitimately hold two
    values (this paper's detector table puts Youden's J and MCC in one).

    Parsing cells beats regexing the row, because the same value may be wrapped in
    ``\\textbf{}``, ``$\\mathbf{}$`` or neither, and two tables in this paper both have rows
    beginning ``$L_1$ &``. `header_marker` disambiguates by table; a missing table or row
    raises rather than returning something plausible.
    """
    start = tex.find(header_marker)
    if start < 0:
        raise LookupError(f"no table with header marker {header_marker!r}")
    body = tex[start : tex.find(r"\bottomrule", start)]
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().startswith(row_label):
            continue
        # A LaTeX row ends at `\\`, which may be several source lines later. Joining to that
        # terminator matters: tab:ciexcl wraps its last column onto a continuation line, and a
        # line-based reader silently returned a SHORT row -- so a checker asking for that column
        # got an IndexError at best and, if it guarded, skipped the cell entirely.
        joined = line
        while "\\\\" not in joined and i + 1 < len(lines):
            i += 1
            joined += " " + lines[i]
        return [
            [float(v) for v in re.findall(r"[+-]?\d+\.\d+|(?<![_\d])[+-]?\d+(?!\d)", cell)]
            for cell in joined.split("&")
        ]
    raise LookupError(f"no row {row_label!r} under {header_marker!r}")


def _scrape(tex: str, pattern: str) -> float:
    """Pull one number out of the paper's own prose.

    Returns NaN when the pattern does not match, which fails the check rather than
    skipping it. That direction matters: a reworded sentence must break the check loudly,
    because a check that silently stops looking is exactly the failure this script exists
    to prevent (see the vacuous integrity scan in the module docstring).
    """
    # MULTILINE so a pattern can anchor a LaTeX table row with ^; DOTALL so one can span
    # the line wraps the .tex is hard-wrapped at.
    m = re.search(pattern, tex, re.DOTALL | re.MULTILINE)
    if m is None:
        return float("nan")
    return float(m.group(1))


def _arm_mean(name: str, arm: str, field: str) -> float:
    return mean(r[arm][field] for r in _runs(name))


#: Number words the paper spells out. Scraping "seventeen" as 17 keeps the prose readable
#: while still making the count a checked claim; anything not listed raises rather than
#: quietly returning NaN, so a new spelled-out number cannot slip past unchecked.
_WORD_NUM = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "eight": 8,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "sixteen": 16,
    "fifteen": 15,
    "seventeen": 17,
    "twenty": 20,
}


def _scrape_word(tex: str, pattern: str) -> float:
    """Like `_scrape`, for a count the paper spells out in words rather than digits."""
    m = re.search(pattern, tex, re.DOTALL | re.MULTILINE)
    if m is None:
        return float("nan")
    word = m.group(1).lower()
    if word not in _WORD_NUM:
        raise LookupError(
            f"the paper spells a count as {word!r}, which is not in _WORD_NUM; add it rather "
            "than letting the check pass on an unparsed word"
        )
    return float(_WORD_NUM[word])


#: The split sweep's rows, as (paper row label, released run slug). Run 3 is the n_train=8 row:
#: it was not bought for the sweep, it *is* the cell the sweep was designed to interrogate.
_SPLIT_ROWS = (("4", "split4"), ("8", "c2b"), ("16", "split16"), ("24", "split24"))

#: The order `sec:threats` lists the five seed-genome replicate means in. Pinned as an explicit
#: order rather than sorted, because the prose reads as a sentence and its order is editorial.
_NOISE_REPLICATE_ORDER = ("c2", "c2b", "split4", "split16", "split24")


def _noise_estimate():
    """The measurement-noise estimator, imported lazily like the other analyzers."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_measurement_noise as mod

    return mod.noise_estimate()


#: The seed-baseline curve's runs, which all score the SAME thirty held-out tasks. Verified on
#: the cassettes rather than asserted: identical prompts hash to identical content-addressed
#: keys, so a shared key IS a shared task. Never trust the metas here -- the older ones do not
#: even record `task_set`.
_CURVE_RUNS = {
    "run3": "multihard",
    "run4": "feedback",
    "run7": "oracle",
    "run8": "fixbudget",
    "run9": "fbactlo",
    "run10": "fbacthi",
}
_CURVE_SHARED_SEEDS = 3


def _shared_first_draft_calls() -> int:
    """Worker calls common to every run of the curve. Should be 3 x (10 heldout + 8 train).

    This is the licence for omitting a task-draw term from the between-run comparisons: if the
    prompts are byte-identical then the tasks are identical and difficulty cancels. If a future
    edit breaks it, the paired standard errors in sec:threats become anti-conservative, so it is
    checked rather than remembered.
    """
    from pathlib import Path

    shared: set[str] | None = None
    for slug in _CURVE_RUNS.values():
        entries = json.loads(Path(f"tests/fixtures/cassettes/study-{slug}.json").read_text())[
            "entries"
        ]
        keys = {k for k, v in entries.items() if v.get("name") == "worker"}
        shared = keys if shared is None else (shared & keys)
    return len(shared or ())


def _curve_seed_mean(label: str, *, shared_seeds_only: bool = True) -> float:
    """Arm A's seed-genome held-out mean, over the seeds the curve's runs share.

    Run 3 has five seeds where the others have three, and seeds 3-4 draw DIFFERENT tasks. Pooling
    all five would silently reintroduce the task-draw variance the paired comparison exists to
    remove -- an error made once while writing this section.
    """
    runs = _runs(f"validation-study-{_CURVE_RUNS[label]}.json")
    if shared_seeds_only:
        runs = runs[:_CURVE_SHARED_SEEDS]
    return mean(r["A"]["seed_heldout"] for r in runs)


def _curve_gap(a: str, b: str) -> tuple[float, float]:
    """(mean gap b-a, gap in units of sigma(Delta)) paired on the shared held-out tasks."""
    gap = _curve_seed_mean(b) - _curve_seed_mean(a)
    return gap, abs(gap) / _noise_estimate().sigma_delta_mean


def _bcb_actionability_overlap() -> float:
    """Measured held-out overlap across the code-actionability run's six seeds.

    In the checks because the paper now declines to report an interval for that effect ON THE
    STRENGTH OF THIS NUMBER. If the overlap were recomputed as zero, the refusal would be
    unmotivated and the paragraph would need rewriting rather than silently standing.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_deltas as ad

    path = f"{DOCS}/validation-study-bcb-acthi.json"
    meta = ad._meta(path)
    task_set = ad.task_set_of(meta)
    return ad.heldout_overlap(task_set, list(meta.get("seeds") or []), meta.get("n_heldout"))


def _detector_stats(cassette: str, seeds: list[int], loop_kw: dict) -> dict[str, float]:
    """Detector precision/F1/n replayed FROM THE CASSETTE, not read from a results JSON.

    This is the paper's second independent derivation path. The seed-held-out column of
    tab:curve comes from the results artifacts; its detector columns come from replaying the
    recorded critique verdicts. Agreement of the two against the same table is a stronger
    statement than either alone.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_critique_detector as acd

    events = acd.collect_events(f"tests/fixtures/cassettes/{cassette}", seeds, loop_kw=loop_kw)
    return acd.detector_stats(events)


def _residual_rates() -> dict[str, dict[str, float]]:
    """Per-constraint final-draft satisfaction, replayed from the cassettes.

    tab:residual's caption says "from cassettes", and this makes that literal: all 24 cells are
    recomputed through the production self-refine loop in strict replay mode. Previously the
    run-3 column existed here only as a hand-typed dict, which pinned nothing about the paper.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_residual_by_constraint as arc

    out: dict[str, dict[str, float]] = {}
    for tag, name, seeds, oc, lvl, r in arc._REGIMES:
        counts = arc.constraint_pass_counts(
            f"{arc.CASSETTE_DIR}/{name}",
            seeds,
            oracle_critique=oc,
            feedback_level=lvl,
            max_retries=r,
        )
        key = tag.split()[0]
        out[key] = {
            iid: arc.pass_rate(counts, iid) for iid in arc._IDS if counts.get(iid, [0, 0])[1]
        }
    return out


def _evolved_conditions():
    """The arm-D mechanism table's rows: seed vs evolved critique scored as detectors.

    Replayed through each condition's own genome against the c2b cassette, which is the only
    comparison the recording supports -- `generate` and `ensure_correct` moved together, so a
    same-drafts contrast was never recorded.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_evolved_gene as ev

    return ev.score_run(
        f"{DOCS}/validation-study-c2b.json", "tests/fixtures/cassettes/study-c2b.json"
    )


def _per_seed_deltas(kind: str) -> list[float]:
    """The per-seed Delta vectors behind tab:ciexcl, recomputed from the runs."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_deltas as ad

    if kind == "actuator":
        runs = _runs("validation-study-bcb-actuator-big.json")
        return [r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs]
    if kind == "actionability":
        return ad.delta_between(
            f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
        )
    runs = _runs("validation-study-multihard.json")
    return [r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs]


def _corrected_ci(kind: str) -> tuple[float, float]:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_deltas as ad

    _, lo, hi = ad.exact_resample_ci(_per_seed_deltas(kind))
    return lo, hi


def _repeat():
    """The D7 replicate driver, imported lazily like the other analyzers."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repeat_scores as mod

    return mod


def _repeat_results() -> dict[str, object]:
    mod = _repeat()
    return {r.level: r for r in mod.aggregate(5)}


def _repeat_sigma() -> dict[tuple[str, str], float]:
    return _repeat().sigma_by_genome(5)


#: The pre-registered budget sweep: (mmc, proposal cap). n_train is fixed at 8 throughout, which
#: is what makes it a single-knob axis where the cap does not bind.
BUDGET_SWEEP = ((20, 6), (28, 6), (40, 6), (56, 6), (80, 6), (160, 12))
BUDGET_N_TRAIN = 8


def _budget_cell(mmc: int, cap: int) -> dict[str, float]:
    """Cycles, effective depth and priced bound for one budget-sweep cell.

    Derived, not transcribed. The bound comes from the same cost model the pre-spend guard uses,
    so the paper's priced column and the guard's refusal threshold cannot disagree -- and the
    whole table is checkable BEFORE the runs exist, which is what a pre-registration needs.
    """
    from whystuck.eval.cost import WORKER_TOKENS, plan_study_calls

    cycles = (mmc - BUDGET_N_TRAIN) / (min(3, BUDGET_N_TRAIN) + BUDGET_N_TRAIN)
    groups = plan_study_calls(
        arms=("B", "D"),
        n_train=BUDGET_N_TRAIN,
        n_heldout=10,
        max_metric_calls=mmc,
        reflection_calls=cap,
        worker_model="openai/gpt-4.1-mini",
        reflection_model="openai/gpt-5.6-luna",
        n_seeds=3,
        worker_calls_per_task=3,
        worker_tokens=WORKER_TOKENS,
    )
    return {
        "cycles": cycles,
        "depth": min(cycles, float(cap)),
        "usd": sum(g.usd() for g in groups),
    }


def _contrast_flag_census() -> tuple[int, int]:
    """(one-sided, both-arms-starved) among released runs with at least one optimizing arm.

    Evidence 9's distinction, computed rather than transcribed. A run where exactly one
    optimizing arm accepted nothing is the new defect: its Delta is labelled D-vs-B and is
    really D-vs-seed. A run where BOTH accepted nothing is the original retraction, visible in
    the scores for the first time. Collapsing the two would let the paper claim the sweep
    surfaced sixteen new defects when it surfaced one.
    """
    import glob
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_deltas as ad

    one_sided = both = 0
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        counts = ad.arms_optimized(path)
        if not counts or ad.is_valid_contrast(path):
            continue
        if any(v > 0 for v in counts.values()):
            one_sided += 1
        else:
            both += 1
    return one_sided, both


@dataclass(frozen=True)
class SplitRow:
    """One row of the split sweep, computed from the released run rather than transcribed."""

    n_train: int
    mmc: int
    n_seeds: int
    seed_mean: float
    delta: float
    per_seed: list[float]
    #: Seeds on which the prompt-only arm changed a gene at all. In the table because at
    #: n_train=4 it is zero, which makes that row a D-vs-seed contrast wearing a D-vs-B label
    #: -- the retraction's defect class, entering from the arm that was not under suspicion.
    b_accepted: int

    @property
    def cycles(self) -> float:
        """Promotable accept cycles: the search depth the constant-epoch line does not hold."""
        return (self.mmc - self.n_train) / (min(3, self.n_train) + self.n_train)


def _split_row(slug: str) -> SplitRow:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_deltas as ad

    path = f"{DOCS}/validation-study-{slug}.json"
    doc = json.loads(Path(path).read_text())
    runs, meta = doc["runs"], doc["meta"]
    n_train, mmc = int(meta["n_train"]), int(meta["max_metric_calls"])
    if (n_train, mmc) not in SWEEP:
        raise ValueError(
            f"{path} ran at (n_train={n_train}, mmc={mmc}), which is not a registered cell of "
            f"the pre-registered sweep {SWEEP}. Either the run is not the one the paper's table "
            "describes, or the pre-registration was edited after the fact."
        )
    return SplitRow(
        n_train=n_train,
        mmc=mmc,
        n_seeds=len(runs),
        seed_mean=mean(r["B"]["seed_heldout"] for r in runs),
        delta=mean(r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs),
        per_seed=[r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs],
        b_accepted=ad.arms_optimized(path)["B"],
    )


def _sweep_pass_zero() -> dict[str, int]:
    """Per sweep row, how many arm-D critique genes emit PASS exactly zero times.

    This is the sweep's load-bearing pre-registered prediction, so it is recomputed by
    replaying each released gene against its own cassette rather than read off a table.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_evolved_gene as ev

    out: dict[str, int] = {}
    for _, slug in _SPLIT_ROWS:
        cassette = f"tests/fixtures/cassettes/study-{'c2b' if slug == 'c2b' else slug}.json"
        rows = ev.score_run(f"{DOCS}/validation-study-{slug}.json", cassette)
        out[slug] = sum(1 for r in rows if r.kind == "evolved" and r.pass_rate == 0.0)
    return out


def _decomp():
    """The component-decomposition module, imported lazily like the other analyzers."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_delta_decomposition as mod

    return mod


#: Level names in the order sec:decomp prints them.
_DECOMP_LEVELS = ("L1", "L2", "L3")


def _decomposition(level: int) -> dict[str, float]:
    mod = _decomp()
    d = mod.decompose(mod.LEVELS[level])
    return {
        "delta": d.delta,
        "draft_term": d.draft_term,
        "loop_term": d.loop_term,
        "draft_b": d.draft_b,
        "draft_d": d.draft_d,
    }


def _stratum(level: int, correct: bool) -> dict[str, float]:
    mod = _decomp()
    g = mod.strata(mod.LEVELS[level])[correct]
    return {
        "n": g.n,
        "mean_b": g.mean_b,
        "mean_d": g.mean_d,
        "gap": g.gap,
        "contribution": g.contribution,
    }


def _refine(level: int, arm: str) -> tuple[int, int, float]:
    mod = _decomp()
    return mod.refine_productivity(mod.LEVELS[level], arm)


def _length(level: int, arm: str) -> tuple[float, float]:
    mod = _decomp()
    return mod.answer_length(mod.LEVELS[level], arm)


def _arm_confusion(level: int, arm: str) -> dict[str, int]:
    """The confusion matrix for one arm's own critique on the drafts its own genome made.

    Arm B evolves only `generate`, so its critique is the seed rubric; arm D's is the
    evolved one. This is the comparator the mechanism table needed: "precision-1
    acceptance" is only a claim if something else in the study accepts imprecisely.
    """
    import json as _json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_critique_detector import collect_events, detector_stats

    mod = _decomp()
    lvl = mod.LEVELS[level]
    runs = _json.loads(Path(lvl.results).read_text())["runs"]
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
    for seed, run in enumerate(runs):
        stats = detector_stats(
            collect_events(
                lvl.cassette,
                [seed],
                loop_kw={},
                gene_overrides=dict(run[arm]["final_genes"]),
                exclude=lvl.exclude,
            )
        )
        for key in total:
            total[key] += int(stats[key])
    return total


def _c1_decomposition() -> tuple[float, float]:
    """(contribution of the eight tasks sec:analysis names, residual on the other 24).

    The paper used to present these two as a decomposition of the whole -0.033. They are
    not: three tasks differ in verdict and five had an already-compliant draft, and the
    other twenty-four moved because the repair *prose* differed, not the verdict. Deriving
    both halves here is what keeps the partial decomposition labelled as partial.
    """
    import json as _json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_critique_detector import collect_events

    from whystuck.eval.multi_constraint import load_multi_constraint

    base_doc = _json.loads(Path(f"{DOCS}/validation-study-c1-base.json").read_text())
    hand_doc = _json.loads(Path(f"{DOCS}/validation-study-c1-hand.json").read_text())
    b_by_task = base_doc["runs"][0]["A"]["seed_heldout_by_task"]
    h_by_task = hand_doc["runs"][0]["A"]["seed_heldout_by_task"]
    n_heldout = len(b_by_task)
    tasks = load_multi_constraint(
        "heldout", seed=0, n=n_heldout, pool=max(64, 2 * n_heldout + 2), hard=True
    )
    # Both conditions share one cassette, so each task's first draft is byte-identical and
    # the verdicts are the only thing that can differ. The gene each condition actually ran
    # is recorded in `meta.gene_overrides` -- empty for the seed condition, which is exactly
    # what "no override" means -- rather than in a per-arm field these arm-A runs never fill.
    ev_seed, ev_hand = (
        collect_events(
            "tests/fixtures/cassettes/study-c1.json",
            [0],
            loop_kw={},
            n_heldout=n_heldout,
            gene_overrides=doc["meta"]["gene_overrides"] or None,
        )
        for doc in (base_doc, hand_doc)
    )
    if not (len(ev_seed) == len(ev_hand) == len(tasks)):
        raise ValueError(
            f"C1 replay produced {len(ev_seed)}/{len(ev_hand)} events for {len(tasks)} tasks; "
            "a decomposition over a partial replay would silently reweight the strata"
        )
    named = 0.0
    for task, (v_seed, actual), (v_hand, _) in zip(tasks, ev_seed, ev_hand, strict=True):
        if task.id not in b_by_task:
            continue
        diff = h_by_task[task.id] - b_by_task[task.id]
        # The eight the paper names: verdicts differ, or the draft was already compliant.
        if v_seed != v_hand or actual:
            named += diff
    total = sum(h_by_task[k] - b_by_task[k] for k in b_by_task)
    return named / n_heldout, (total - named) / n_heldout


def _reflection_models() -> tuple[set[str], set[str]]:
    """(reflection models on corrected-harness runs, on pre-correction runs).

    The paper's Setup paragraph named one model for the whole study, and the corrected runs
    used the other: `gpt-5.6-luna` is not the CLI default, and every run the conclusions rest
    on passed it explicitly. A single-model claim about a two-model study is the kind of thing
    a reader checks against the released metadata, so it is checked here.

    Only runs with an evolving arm count. An `--arms A` run records whatever
    `--reflection-model` defaulted to, but never reflects, so its metadata names a model that
    did not run -- counting it would report a two-model study where the field is simply inert.
    """
    import glob

    corrected: set[str] = set()
    pre: set[str] = set()
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        model = doc["meta"].get("reflection_model")
        first = doc["runs"][0]
        arms = {k for k in first if isinstance(first.get(k), dict)}
        if model is None or not (arms & {"B", "D"}):
            continue
        (corrected if doc["meta"].get("module_selector") == "all" else pre).add(str(model))
    return corrected, pre


def _precorrection_counts() -> tuple[int, int, float]:
    """(contrast runs recorded pre-correction, how many have Δ exactly 0, largest |mean Δ|).

    `meta.module_selector == "all"` is the marker the corrected harness writes, so this
    partitions the released runs by harness generation without a hand-maintained list. It
    exists because the paper said "eleven" for a set that is fifteen, and separately claimed
    every Δ in it was an identity when four are not: an unchanged verify gene leaves the arms
    differing through the `generate` gene they share, which the default selector made arm D
    search half as hard (Evidence 3).
    """
    import glob
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_deltas import delta_arm

    means = []
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        first = doc["runs"][0]
        arms = sorted(k for k in first if isinstance(first.get(k), dict))
        if arms != ["A", "B", "C", "D"] or doc["meta"].get("module_selector") == "all":
            continue
        means.append(mean(delta_arm(path)))
    if not means:
        raise ValueError(
            "no pre-correction contrast runs found; the module_selector marker must have "
            "changed, and a count of zero would silently retract the paper's own caution"
        )
    return len(means), sum(1 for m in means if m == 0.0), max(abs(m) for m in means)


def _registry_counts() -> tuple[int, int]:
    """(superseded, replayable) as `verify_reproducibility.py` itself defines them."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import verify_reproducibility as vr

    return len(vr.SUPERSEDED), vr.EXPECTED_RUNS


#: (claim, expected, tolerance, thunk). Tolerance matches the precision the paper prints at.
def _checks() -> list[tuple[str, float, float, float]]:
    out: list[tuple[str, float, float, float]] = []

    # --- Table "detector quality vs loop performance" (the non-monotonicity claim) -----
    for label, f, claimed in (
        ("tab:curve run3 intrinsic seed", "validation-study-multihard.json", 0.765),
        ("tab:curve run4 tool-fed seed", "validation-study-feedback.json", 0.837),
        ("tab:curve run7 oracle mid r1", "validation-study-oracle.json", 0.748),
        ("tab:curve run8 oracle mid r3", "validation-study-fixbudget.json", 0.813),
        ("tab:curve run9 oracle lo r3", "validation-study-fbactlo.json", 0.810),
        ("tab:curve run10 oracle hi r3", "validation-study-fbacthi.json", 0.873),
    ):
        out.append((label, claimed, 0.0005, _arm_mean(f, "A", "seed_heldout")))

    # --- Evidence 3b: the per-seed actuator decomposition -----------------------------
    act = _runs("validation-study-bcb-actuator-big.json")
    for i, (bs, bo, ds, do) in enumerate(
        [(0.55, 0.60, 0.55, 0.45), (0.45, 0.60, 0.45, 0.45), (0.65, 0.65, 0.65, 0.65)]
    ):
        for arm, cs, co in (("B", bs, bo), ("D", ds, do)):
            out.append((f"ev3b seed{i} {arm} seed", cs, 0.005, act[i][arm]["seed_heldout"]))
            out.append((f"ev3b seed{i} {arm} opt", co, 0.005, act[i][arm]["opt_heldout"]))

    # --- The worker-swap replication (run 11) -----------------------------------------
    nano = _runs("validation-study-multihard-nano.json")
    out.append(("run11 arm B mean", 0.70, 0.005, mean(r["B"]["opt_heldout"] for r in nano)))
    out.append(("run11 arm B std", 0.03, 0.005, pstdev([r["B"]["opt_heldout"] for r in nano])))
    out.append(("run11 arm D mean", 0.69, 0.005, mean(r["D"]["opt_heldout"] for r in nano)))
    out.append(("run11 arm D std", 0.03, 0.005, pstdev([r["D"]["opt_heldout"] for r in nano])))
    out.append(
        (
            "run11 seed",
            0.67,
            0.005,
            _arm_mean("validation-study-multihard-nano.json", "A", "seed_heldout"),
        )
    )

    # --- Headline effects and intervals -----------------------------------------------
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_c1 import paired_by_task
    from analyze_deltas import delta_arm, delta_between, exact_resample_ci, sign_test_p

    d_act = delta_arm(f"{DOCS}/validation-study-bcb-actuator-big.json")
    m, lo, hi = exact_resample_ci(d_act)
    out.append(("actuator delta(D-B)", -0.10, 0.005, m))
    out.append(("actuator CI lo", -0.150, 0.0005, lo))
    out.append(("actuator CI hi", 0.000, 0.0005, hi))
    out.append(("actuator sign p", 0.50, 0.005, sign_test_p(d_act)))

    d_hi = delta_between(
        f"{DOCS}/validation-study-fbacthi.json", f"{DOCS}/validation-study-fbactlo.json"
    )
    m, lo, hi = exact_resample_ci(d_hi)
    out.append(("actionability delta(hi-lo)", 0.063, 0.0005, m))
    out.append(("actionability CI lo", 0.025, 0.0005, lo))
    out.append(("actionability CI hi", 0.090, 0.0005, hi))
    out.append(("actionability sign p", 0.25, 0.005, sign_test_p(d_hi)))

    d_code = delta_between(
        f"{DOCS}/validation-study-bcb-acthi.json", f"{DOCS}/validation-study-bcb-actlo.json"
    )
    out.append(("code actionability delta", 0.200, 0.0005, exact_resample_ci(d_code)[0]))

    d_n3 = delta_arm(f"{DOCS}/validation-study-multihard.json")
    out.append(("run3 delta(D-B)", 0.00, 0.0005, exact_resample_ci(d_n3)[0]))
    d_n11 = delta_arm(f"{DOCS}/validation-study-multihard-nano.json")
    out.append(("run11 delta(D-B)", -0.014, 0.0005, exact_resample_ci(d_n11)[0]))

    # --- The C1 verifier ablation (Analysis 1d) ---------------------------------------
    c1 = list(
        paired_by_task(
            f"{DOCS}/validation-study-c1-base.json", f"{DOCS}/validation-study-c1-hand.json"
        ).values()
    )
    out.append(("C1 n tasks", 32, 0, len(c1)))
    out.append(("C1 ties", 17, 0, sum(1 for d in c1 if d == 0.0)))
    out.append(("C1 hand better", 6, 0, sum(1 for d in c1 if d > 0)))
    out.append(("C1 seed better", 9, 0, sum(1 for d in c1 if d < 0)))
    out.append(("C1 mean delta", -0.033, 0.0005, mean(c1)))
    out.append(("C1 sign p", 0.61, 0.005, sign_test_p(c1)))

    # --- The corrected-harness re-run, C2 (tab:rerun and the resolved prediction) ------
    # The gene-change counts are the load-bearing ones: they are what distinguishes this
    # run from the eleven it replaces, and the claim "the treatment was applied" reduces
    # entirely to them. Checked from the released JSON rather than from the prose.
    c2 = _runs("validation-study-c2.json")
    out.append(("C2 seeds", 3, 0, len(c2)))
    for arm, seed_m, opt_m in (
        ("A", 0.790, 0.790),
        ("C", 0.790, 0.790),
        ("B", 0.790, 0.750),
        ("D", 0.790, 0.735),
    ):
        out.append(
            (f"C2 {arm} seed mean", seed_m, 0.0005, mean(r[arm]["seed_heldout"] for r in c2))
        )
        out.append((f"C2 {arm} opt mean", opt_m, 0.0005, mean(r[arm]["opt_heldout"] for r in c2)))
    out.append(
        (
            "C2 arm D evolved ensure_correct on N seeds",
            2,
            0,
            sum(1 for r in c2 if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        )
    )
    out.append(
        (
            "C2 arm C changed nothing on N seeds",
            3,
            0,
            sum(1 for r in c2 if not (r["C"].get("genes_changed") or [])),
        )
    )
    d_c2 = delta_arm(f"{DOCS}/validation-study-c2.json")
    m, lo, hi = exact_resample_ci(d_c2)
    out.append(("C2 delta(D-B)", -0.015, 0.0005, m))
    out.append(("C2 CI lo", -0.065, 0.0005, lo))
    out.append(("C2 CI hi", 0.020, 0.0005, hi))
    # The per-seed drops the overfitting mechanism is stated on: seed 2 is where arm D
    # accepted two candidates against arm B's one and fell furthest, and seed 0 is where
    # no arm accepted anything. The acceptance counts themselves come from the run log
    # and are NOT in the released JSON, so only the scores they explain are checked here.
    # Evidence 8: the seed-2 arm-D critique gene is EMPTY, and two reflection completions
    # were empty. Pinned because the paper's "the treatment was applied" claim is qualified
    # entirely by these two numbers, and because the check that should have caught it
    # (scanning the cassette for empty outputs) was itself vacuous until it was rewritten.
    out.append(
        ("C2 seed2 D ensure_correct length", 0, 0, len(c2[2]["D"]["final_genes"]["ensure_correct"]))
    )
    out.append(
        (
            "C2 seed1 D ensure_correct length",
            3101,
            0,
            len(c2[1]["D"]["final_genes"]["ensure_correct"]),
        )
    )
    with open("tests/fixtures/cassettes/study-c2.json", encoding="utf-8") as fh:
        entries = json.load(fh)["entries"]
    vals = list(entries.values()) if isinstance(entries, dict) else entries
    refl = [v["output"] for v in vals if v["name"] == "reflection"]
    out.append(("C2 empty reflection completions", 2, 0, sum(1 for o in refl if not o.strip())))
    out.append(
        (
            "C2 empty worker completions",
            0,
            0,
            sum(1 for v in vals if v["name"] == "worker" and not v["output"].strip()),
        )
    )
    out.append(("C2 seed2 B opt", 0.750, 0.0005, c2[2]["B"]["opt_heldout"]))
    out.append(("C2 seed2 D opt", 0.685, 0.0005, c2[2]["D"]["opt_heldout"]))

    # --- C2b: the CLEAN re-run, which is what tab:rerun and the conclusion report --------
    # C2 above is retained only as the evidence for Evidence 8. Everything the paper claims
    # about the corrected harness -- treatment applied on 3/3 seeds, prompt-only flat,
    # loop-level costing 0.042 -- is checked here against the clean run.
    c2b = _runs("validation-study-c2b.json")
    out.append(("C2b seeds", 3, 0, len(c2b)))
    for arm, opt_m in (("A", 0.788), ("C", 0.788), ("B", 0.788), ("D", 0.747)):
        out.append(
            (f"C2b {arm} seed mean", 0.788, 0.0005, mean(r[arm]["seed_heldout"] for r in c2b))
        )
        out.append((f"C2b {arm} opt mean", opt_m, 0.0005, mean(r[arm]["opt_heldout"] for r in c2b)))
    out.append(
        (
            "C2b arm D evolved ensure_correct on N seeds",
            3,
            0,
            sum(1 for r in c2b if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        )
    )
    out.append(
        (
            "C2b arm C changed nothing on N seeds",
            3,
            0,
            sum(1 for r in c2b if not (r["C"].get("genes_changed") or [])),
        )
    )
    # The evolved rubrics are substantial, against a 127-char seed -- the contrast with C2's
    # zero-length gene is the whole point of the re-run.
    for i, n in enumerate((3235, 3022, 2920)):
        out.append(
            (
                f"C2b seed{i} D ensure_correct length",
                n,
                0,
                len(c2b[i]["D"]["final_genes"]["ensure_correct"]),
            )
        )
    out.append(("C2b seed gene length", 127, 0, len(c2b[0]["C"]["final_genes"]["ensure_correct"])))
    d_c2b = delta_arm(f"{DOCS}/validation-study-c2b.json")
    m, lo, hi = exact_resample_ci(d_c2b)
    out.append(("C2b delta(D-B)", -0.042, 0.0005, m))
    out.append(("C2b CI lo", -0.095, 0.0005, lo))
    out.append(("C2b CI hi", 0.050, 0.0005, hi))
    out.append(("C2b sign p", 1.00, 0.005, sign_test_p(d_c2b)))
    # No empty completion anywhere: the defect Evidence 8 describes did not recur.
    with open("tests/fixtures/cassettes/study-c2b.json", encoding="utf-8") as fh:
        b_entries = json.load(fh)["entries"]
    b_vals = list(b_entries.values()) if isinstance(b_entries, dict) else b_entries
    out.append(("C2b empty completions", 0, 0, sum(1 for v in b_vals if not v["output"].strip())))
    # The evolved critique scored as a detector: PASS emitted zero times on every seed, and
    # pooled true positives 0 against the seed rubric's 1. This is the paper's mechanism for
    # arm D's loss, so it is checked rather than remembered. Replaying is slow (three
    # cassette passes), so only the two load-bearing integers are pinned here; the full
    # table comes from scripts/analyze_evolved_gene.py.
    out.append(
        (
            "C2b evolved rubric lengths all > 2500",
            3,
            0,
            sum(1 for r in c2b if len(r["D"]["final_genes"]["ensure_correct"]) > 2500),
        )
    )
    out.append(
        (
            "C2b shortest reflection output",
            968,
            0,
            min(len(v["output"]) for v in b_vals if v["name"] == "reflection"),
        )
    )

    # --- The metric's do-nothing floor (Threats) --------------------------------------
    from whystuck.eval.ifbench import ifbench_score, load_ifbench
    from whystuck.eval.multi_constraint import load_multi_constraint, mc_score

    ifb = load_ifbench("all")
    mh = load_multi_constraint("all", pool=64, hard=True)
    out.append(("floor ifbench empty", 0.548, 0.0005, mean([ifbench_score("", t) for t in ifb])))
    out.append(("floor ifbench 'x'", 0.452, 0.0005, mean([ifbench_score("x", t) for t in ifb])))
    out.append(("floor multihard empty", 0.420, 0.0005, mean([mc_score("", t) for t in mh])))
    out.append(("floor multihard 'x'", 0.499, 0.0005, mean([mc_score("x", t) for t in mh])))
    out.append(("ifbench pool size", 62, 0, len(ifb)))

    # --- The oracle taxonomy's verdict-equivalence count ------------------------------
    verdict_equivalent = {"exact_word_count", "word_max_length", "end_with", "num_bullets"}
    inst = [i for t in mh for i in t.payload["instruction_id_list"]]
    out.append(("taxonomy instances total", 288, 0, len(inst)))
    out.append(
        ("taxonomy verdict-equivalent", 165, 0, sum(1 for i in inst if i in verdict_equivalent))
    )
    out.append(("taxonomy families total", 8, 0, len(set(inst))))
    out.append(("taxonomy hard-pool tasks", 64, 0, len(mh)))

    # --- Cost overrides: how many runs raised --max-usd above the $5 default ----------
    import glob

    est = []
    for f in glob.glob(f"{DOCS}/validation-study*.json"):
        with open(f, encoding="utf-8") as fh:
            est.append(json.load(fh)["meta"].get("estimated_usd", 0))
    out.append(("runs over the $5 default", 4, 0, sum(1 for u in est if u > 5.0)))
    out.append(("largest estimate", 11.76, 0.005, max(est)))

    # Prose cross-checks. Everything above re-derives a number from the artifacts and
    # compares it to a number typed into *this file*; nothing above reads the paper. That
    # gap let a real error through: the closability table said the re-run changed
    # ensure_correct on 2/3 seeds when the data (and the body text) say 3/3 -- the 2/3 is
    # arm B's generate, transposed. The checks below scrape the paper's own sentences, so
    # the claimed side comes from main.tex rather than from an author's memory of it.
    tex = _tex()
    d_row = r"D & \\texttt\{generate\} \$\+\$ \\texttt\{ensure\\_correct\} & 0\.788 & "
    prose: list[tuple[str, str, float, float]] = [
        (
            "tex: D evolved ensure_correct on N/3",
            r"corrected harness changed \\texttt\{ensure\\_correct\} on \$(\d)/3\$",
            0,
            sum(1 for r in c2b if "ensure_correct" in (r["D"].get("genes_changed") or [])),
        ),
        (
            "tex: tab:rerun D optimized",
            d_row + r"([\d.]+) &",
            0.0005,
            mean(r["D"]["opt_heldout"] for r in c2b),
        ),
        (
            "tex: tab:rerun D delta",
            d_row + r"[\d.]+ & \$\\mathbf\{(-?[\d.]+)\}",
            0.0005,
            mean(r["D"]["opt_heldout"] - r["D"]["seed_heldout"] for r in c2b),
        ),
        (
            "tex: shortest reflection output",
            r"shortest reflection output: \$(\d+)\$",
            0,
            968,
        ),
        (
            "tex: seed rubric length",
            r"against a \$(\d+)\$-character seed",
            0,
            len(c2b[0]["C"]["final_genes"]["ensure_correct"]),
        ),
    ]
    for i in range(3):
        prose.append(
            (
                f"tex: evolved rubric {i} length",
                r"detector rubrics of " + r"\$\d+\$,?\s*(?:and\s*)?" * i + r"\$(\d+)\$",
                0,
                len(c2b[i]["D"]["final_genes"]["ensure_correct"]),
            )
        )
        prose.append(
            (
                f"tex: per-seed delta(D-B) {i}",
                r"Per-seed \$\\dd\(D-B\) = \[" + r"-?[\d.+]+, " * i + r"(-?[\d.+]+)",
                0.0005,
                d_c2b[i],
            )
        )
    # The pre-registered split-size sweep (sec:splitsweep). Nothing here is a measurement
    # yet -- these are the sweep's DESIGN numbers, and pinning them is the point: the
    # prediction is only pre-registered if the configuration it was written against cannot
    # be edited afterwards without a gate noticing. Every value is re-derived from the
    # generator and from GEPA's documented budget arithmetic, not copied from the table.
    for n_train, mmc in SWEEP:
        row = rf"^{n_train}\s*& {mmc} & (\d+) &"
        prose.append(
            (
                f"tex: sweep n={n_train} accept-cycle floor",
                row,
                0,
                2 * n_train + min(3, n_train),
            )
        )
        prose.append(
            (
                f"tex: sweep n={n_train} epochs (mmc/n)",
                rf"^{n_train}\s*& (\d+) &",
                0.0005,
                round(EPOCHS * n_train),
            )
        )
        # `row` captures the floor, so the overlap patterns need it non-capturing or
        # re.search hands back the floor as group(1).
        row_nc = rf"^{n_train}\s*& {mmc} & \d+ &"
        # Counts, not shares: an exact integer has no display rounding to argue about.
        for i, count in enumerate(_type_overlap_counts(n_train)):
            prose.append(
                (
                    f"tex: sweep n={n_train} overlap seed{i}",
                    row_nc + r" " + r"\d+ / " * i + r"(\d+)",
                    0,
                    count,
                )
            )
    # The pre-registered p0 sweep (sec:p0). A prediction is only pre-registered if the
    # configuration it was written against is pinned, so E[p0] is re-derived from the generator
    # and the residual table rather than copied from the table.
    #
    # Located by table header, NOT by regex: the detector table of sec:detector also has rows
    # beginning "$L_1$ &", so a bare pattern matched whichever table came first in the file and
    # silently compared E[p0] against an F1. Two tables sharing a row label is exactly the kind
    # of coincidence a positional regex cannot see.
    sweep_marker = r"Level & excluded &"
    for level, (excl, e_p0, max_p0) in enumerate(_P0_LEVELS, start=1):
        cells = _table_row(tex, sweep_marker, rf"$L_{level}$")
        got_e, got_max = _expected_p0(excl)
        runs = _runs(_P0_RUN_FILES[level - 1])
        # Columns: level, exclusions (prose), E[p0], measured p0, B seed->opt, D seed->opt,
        # delta(D-B), 95% CI.
        out.append((f"tex: p0 L{level} E[p0]", cells[2][0], 0.0005, got_e))
        out.append(
            (
                f"tex: p0 L{level} measured p0",
                cells[3][0],
                0.0005,
                _detector_metrics(level - 1)["p0"],
            )
        )
        for pos, arm in ((4, "B"), (5, "D")):
            for j, field in enumerate(("seed_heldout", "opt_heldout")):
                out.append(
                    (
                        f"tex: p0 L{level} {arm} {field.split('_')[0]}",
                        cells[pos][j],
                        0.0005,
                        mean(r[arm][field] for r in runs),
                    )
                )
        deltas = _p0_deltas(level - 1)
        lo, hi = _p0_ci(level - 1)
        out.append((f"tex: p0 L{level} delta(D-B)", cells[6][0], 0.0005, mean(deltas)))
        out.append((f"tex: p0 L{level} CI lo", cells[7][0], 0.0005, lo))
        out.append((f"tex: p0 L{level} CI hi", cells[7][1], 0.0005, hi))
        assert abs(e_p0 - got_e) < 0.0005 and abs(max_p0 - got_max) < 0.005, (
            f"L{level} constants in this file disagree with the generator"
        )
    # The MEASURED p0 at L1, recovered from the C2b cassette: the first critique event judged
    # the first draft, so the ground truth of that draft is p0 by definition. Pinned because
    # the paper uses it to state that the independence estimate runs ~2x high, and a claim
    # about an estimator's bias must not drift from the measurement that established it.
    prose.append(
        (
            "tex: p0 L1 measured numerator",
            r"the measured \$p_0\$\s+is \$(\d+)/\d+ = [\d.]+\$",
            0,
            _measured_p0_l1()[0],
        )
    )
    prose.append(
        (
            "tex: p0 L1 measured denominator",
            r"the measured \$p_0\$\s+is \$\d+/(\d+) = [\d.]+\$",
            0,
            _measured_p0_l1()[1],
        )
    )
    prose.append(
        (
            "tex: p0 L1 measured value",
            r"the measured \$p_0\$\s+is \$\d+/\d+ = ([\d.]+)\$",
            0.0005,
            _measured_p0_l1()[0] / _measured_p0_l1()[1],
        )
    )
    prose.append(("tex: sweep pool below cliff", r"hence \$(\d+)\$ for every", 0, 64))
    prose.append(
        (
            "tex: sweep highest safe n_train",
            r"n_\{\\mathrm\{train\}\} \\le (\d+)\$ and larger",
            0,
            POOL_CLIFF - 1,
        )
    )
    prose.append(("tex: sweep pool cliff", r"and larger from \$(\d+)\$", 0, POOL_CLIFF))

    for label, pattern, tol, artifact in prose:
        out.append((label, _scrape(tex, pattern), tol, artifact))

    # The detector-vs-p0 table, every cell. An unchecked number in a table is exactly the gap
    # that let a transposed seed count survive several readings.
    marker = r"level & $p_0$ & $\mathrm{F}_1$"
    for i, (label, _, _, _) in enumerate(_P0_DETECTOR):
        m = _detector_metrics(i)
        cells = _table_row(tex, marker, rf"$L_{i + 1}$")
        # Columns: level, p0, F1, sensitivity, specificity, then "J / MCC" together.
        for pos, key in ((1, "p0"), (2, "f1"), (3, "sens"), (4, "spec")):
            out.append((f"tex: detector {label} {key}", cells[pos][0], 0.0005, m[key]))
        out.append((f"tex: detector {label} Youden J", cells[5][0], 0.0005, m["j"]))
        out.append((f"tex: detector {label} MCC", cells[5][1], 0.0005, m["mcc"]))

    # The arm-D mechanism table (sec:p0). This table was the paper's least-checked and
    # most load-bearing: it carries the PASS-rate story, and until now nothing re-derived
    # it. It is also the table where a reader recomputing a base rate from tp+fn gets the
    # judged-draft rate rather than p0, so both columns are pinned.
    mech = r"level & critique & base rate judged"
    for level_i, (label, _, _, _) in enumerate(_P0_DETECTOR):
        for arm, arm_label in (("B", "B, seed rubric"), ("D", "D, evolved")):
            counts = _arm_confusion(level_i, arm)
            cells = _table_row(tex, mech, f"$L_{level_i + 1}$ & {arm_label}")
            base = (counts["tp"] + counts["fn"]) / counts["n"]
            out.append((f"tex: mech {label} {arm} base rate", cells[2][0], 0.0005, base))
            out.append(
                (
                    f"tex: mech {label} {arm} PASS emitted",
                    cells[3][0],
                    0,
                    counts["tp"] + counts["fp"],
                )
            )
            for pos, key in ((4, "tp"), (5, "fp"), (6, "fn"), (7, "tn")):
                out.append((f"tex: mech {label} {arm} {key}", cells[pos][0], 0, counts[key]))

    # The component decomposition (sec:decomp): Delta = draft term + loop term. Every cell
    # of both tables, plus the identity that the strata close on the pooled Delta.
    dec_marker = r"level & $\dd(D-B)$ & draft term"
    strat_marker = r"level & stratum & $n$ & B & D"
    for level_i, level in enumerate(_DECOMP_LEVELS):
        d = _decomposition(level_i)
        cells = _table_row(tex, dec_marker, f"$L_{level_i + 1}$")
        out.append((f"tex: decomp {level} delta", cells[1][0], 0.0005, d["delta"]))
        out.append((f"tex: decomp {level} draft term", cells[2][0], 0.0005, d["draft_term"]))
        out.append((f"tex: decomp {level} loop term", cells[3][0], 0.0005, d["loop_term"]))
        for correct, row_label in ((True, "already correct"), (False, "already wrong")):
            g = _stratum(level_i, correct)
            cells = _table_row(tex, strat_marker, f"$L_{level_i + 1}$ & {row_label}")
            out.append((f"tex: strata {level} {row_label} n", cells[2][0], 0, g["n"]))
            out.append((f"tex: strata {level} {row_label} B", cells[3][0], 0.00005, g["mean_b"]))
            out.append((f"tex: strata {level} {row_label} D", cells[4][0], 0.00005, g["mean_d"]))
            out.append((f"tex: strata {level} {row_label} gap", cells[5][0], 0.00005, g["gap"]))
            out.append(
                (
                    f"tex: strata {level} {row_label} contribution",
                    cells[6][0],
                    0.0005,
                    g["contribution"],
                )
            )

    # The prose numbers sec:decomp and sec:threats quote, in the order they are written.
    seed_refine = [_refine(i, "SEED")[2] for i in range(3)]
    final_words = [(_length(i, "B")[1], _length(i, "D")[1]) for i in range(3)]
    decomp_prose: list[tuple[str, str, float, float]] = [
        (
            "tex: decomp L1 overstatement factor",
            r"penalty by a factor of \$([\d.]+)\$",
            0.05,
            abs(_decomposition(0)["delta"]) / abs(_decomposition(0)["loop_term"]),
        ),
        (
            "tex: decomp L2 both arms draft score",
            r"both arms' drafts score \$([\d.]+)\$ there",
            0.0005,
            _decomposition(1)["draft_b"],
        ),
        ("tex: decomp D refines at L1", r"refines\s+\$(\d+)/30\$ held-out", 0, _refine(0, "D")[1]),
        (
            "tex: decomp B refines at L1",
            r"held-out tasks against arm~B's \$(\d+)/30\$",
            0,
            _refine(0, "B")[1],
        ),
        (
            "tex: decomp D gain per refine",
            r"worth less, \$\+([\d.]+)\$ against",
            0.0005,
            _refine(0, "D")[2],
        ),
        (
            "tex: decomp B gain per refine",
            r"worth less, \$\+[\d.]+\$ against \$\+([\d.]+)\$",
            0.0005,
            _refine(0, "B")[2],
        ),
    ]
    for i in range(3):
        decomp_prose.append(
            (
                f"tex: decomp seed refine sign {i}",
                r"on the seed genome \(" + r"\$\+[\d.]+\$,\s+" * i + r"\$\+([\d.]+)\$",
                0.0005,
                seed_refine[i],
            )
        )
    # sec:threats quotes six answer-length figures, D then B at each level.
    length_patterns = (
        (r"never\s+accepts is \$([\d.]+)\$ words", final_words[0][1]),
        (r"against prompt-only's \$([\d.]+)\$ at \$L_1\$", final_words[0][0]),
        (r"\(\$([\d.]+)\$ vs\.\\? \$[\d.]+\$ at \$L_2\$", final_words[1][1]),
        (r"\(\$[\d.]+\$ vs\.\\? \$([\d.]+)\$ at \$L_2\$", final_words[1][0]),
        (r"\$([\d.]+)\$ vs\.\\? \$[\d.]+\$ at \$L_3\$", final_words[2][1]),
        (r"\$[\d.]+\$ vs\.\\? \$([\d.]+)\$ at \$L_3\$", final_words[2][0]),
    )
    for i, (pattern, artifact) in enumerate(length_patterns):
        decomp_prose.append((f"tex: threats answer length {i}", pattern, 0.05, artifact))

    # The C1 decomposition: the eight verdict-differing tasks account for only part of the
    # -0.033, and the paper now says how much. A partial decomposition presented as a whole
    # one is exactly the kind of claim this file exists to keep honest.
    verdict_share, prose_residual = _c1_decomposition()
    decomp_prose.append(
        (
            "tex: C1 eight-task share",
            r"account for \$(-[\d.]+)\$ of the \$-0\.033\$",
            0.0005,
            verdict_share,
        )
    )
    decomp_prose.append(
        (
            "tex: C1 residual on the other 24",
            r"the\nremaining \$(-[\d.]+)\$ falls on the twenty-four",
            0.0005,
            prose_residual,
        )
    )

    for label, pattern, tol, artifact in decomp_prose:
        out.append((label, _scrape(tex, pattern), tol, artifact))

    # --- Table cells, read positionally out of the .tex ------------------------------
    # THE DRIFT HOLE THIS CLOSES. Most checks in this file compare a computed artifact value
    # against a literal typed here, which pins the ARTIFACT but never opens main.tex: edit a
    # number in a table and nothing fails. `_table_row` locates a row by its table's header and
    # its own label, so these assertions read what the paper actually prints, cell by cell.
    # Every table below carries numbers derived from a run rather than from the design.
    c2b_runs = _runs("validation-study-c2b.json")
    seed_mean = mean(r["A"]["seed_heldout"] for r in c2b_runs)
    rerun_hdr = r"Arm & GEPA evolves & seed & optimized"
    for row_label, arm in (("A &", "A"), ("C &", "C"), ("B &", "B"), ("D &", "D")):
        cells = _table_row(tex, rerun_hdr, row_label)
        opt = mean(r[arm]["opt_heldout"] for r in c2b_runs)
        # cells: [arm(no number)] [genes] [seed] [optimized] [delta]
        out.append((f"tab:rerun {arm} seed", cells[2][0], 0.0005, seed_mean))
        out.append((f"tab:rerun {arm} optimized", cells[3][0], 0.0005, opt))
        out.append((f"tab:rerun {arm} delta", cells[4][0], 0.0005, opt - seed_mean))

    p0_hdr = r"Level & excluded & $\mathbb{E}[p_0]$ & $p_0$"
    for row_label, slug in (("$L_1$ &", "c2b"), ("$L_2$ &", "p0-L2"), ("$L_3$ &", "p0-L3")):
        cells = _table_row(tex, p0_hdr, row_label)
        runs = _runs(f"validation-study-{slug}.json")
        b_seed = mean(r["B"]["seed_heldout"] for r in runs)
        b_opt = mean(r["B"]["opt_heldout"] for r in runs)
        d_seed = mean(r["D"]["seed_heldout"] for r in runs)
        d_opt = mean(r["D"]["opt_heldout"] for r in runs)
        lvl = row_label.strip(" &")
        # cells: [Level] [excluded] [E p0] [p0] [B seed->opt] [D seed->opt] [delta] [CI]
        out.append((f"tab:p0 {lvl} B seed", cells[4][0], 0.0005, b_seed))
        out.append((f"tab:p0 {lvl} B opt", cells[4][1], 0.0005, b_opt))
        out.append((f"tab:p0 {lvl} D seed", cells[5][0], 0.0005, d_seed))
        out.append((f"tab:p0 {lvl} D opt", cells[5][1], 0.0005, d_opt))
        out.append((f"tab:p0 {lvl} delta", cells[6][0], 0.0005, d_opt - b_opt))

    decomp_hdr = r"level & $\dd(D-B)$ & draft term"
    for i, row_label in enumerate(("$L_1$ &", "$L_2$ &", "$L_3$ &")):
        cells = _table_row(tex, decomp_hdr, row_label)
        d = _decomposition(i)
        lvl = row_label.strip(" &")
        out.append((f"tab:decomp {lvl} delta", cells[1][0], 0.0005, d["delta"]))
        out.append((f"tab:decomp {lvl} draft", cells[2][0], 0.0005, d["draft_term"]))
        out.append((f"tab:decomp {lvl} loop", cells[3][0], 0.0005, d["loop_term"]))

    # tab:curve's "seed held-out" column, the six-plus-one arm-A baselines that answer
    # questions (i) and (ii). Row labels carry $R$ where two rows share a detector name, since
    # `_table_row` returns the first match and runs 7 and 8 differ only in retries. No
    # try/except here on purpose: an earlier version skipped a wrong header silently and all
    # six checks vanished while the script still printed a pass.
    curve_hdr = r"Critique detector & $R$ & precision"
    #: (row label as printed, results slug, cell index of the seed held-out column)
    curve_rows = (
        ("intrinsic self-critique &", "multihard", 5),
        ("intrinsic $+$ plan module &", "plan", 5),
        ("non-oracle measurement tool &", "feedback", 5),
        ("oracle, which-constraint (mid) & 1", "oracle", 5),
        ("oracle, which-constraint (mid) & 3", "fixbudget", 5),
        ("oracle, bare ``revise'' (lo) &", "fbactlo", 5),
        ("oracle, concrete measured edits (hi) &", "fbacthi", 5),
    )
    for row_label, slug, col in curve_rows:
        cells = _table_row(tex, curve_hdr, row_label)
        artifact = _arm_mean(f"validation-study-{slug}.json", "A", "seed_heldout")
        out.append((f"tab:curve {slug} seed", cells[col][0], 0.0005, artifact))

    # tab:curve's DETECTOR columns, replayed from the cassettes rather than read from a results
    # JSON -- the paper's second derivation path, landing in the same table as the first.
    from whystuck.eval.text_stats import text_stats as _text_stats

    for row_label, cassette, seeds, kw, want_f1 in (
        ("intrinsic self-critique &", "study-multihard.json", [0, 1, 2, 3, 4], {}, True),
        (
            "intrinsic $+$ plan module &",
            "study-plan.json",
            [0, 1, 2],
            {"plan_gene": "plan"},
            False,  # F1 is undefined here (no true positives), and the paper prints "approx 0"
        ),
        (
            "non-oracle measurement tool &",
            "study-feedback.json",
            [0, 1, 2],
            {"stats_fn": _text_stats},
            True,
        ),
    ):
        cells = _table_row(tex, curve_hdr, row_label)
        st = _detector_stats(cassette, seeds, kw)
        tag = row_label.strip(" &").replace("$+$", "plus")
        out.append((f"tab:curve {tag} precision", cells[2][0], 0.005, st["precision"]))
        out.append((f"tab:curve {tag} n", cells[4][0], 0, st["n"]))
        if want_f1:
            out.append((f"tab:curve {tag} F1", cells[3][0], 0.005, st["f1"]))

    # tab:decomp's treated-share column: |loop| / |delta|, the number the section's headline
    # claim ("overstates by 2.5x") is the reciprocal of.
    for i, row_label in (("0", "$L_1$ &"), ("1", "$L_2$ &")):
        d = _decomposition(int(i))
        cells = _table_row(tex, decomp_hdr, row_label)
        out.append(
            (
                f"tab:decomp {row_label.strip(' &')} treated share",
                cells[4][0],
                0.5,
                100.0 * abs(d["loop_term"]) / abs(d["delta"]),
            )
        )

    # tab:splitsweep's priced bounds. These are money figures the prose of sec:splitresult now
    # cites as the bound that was NOT exceeded, so they must come from each run's own recorded
    # pre-flight estimate rather than from memory.
    sweep_hdr = r"$n_{\mathrm{train}}$ & \texttt{mmc} & accept-cycle floor"
    for row_label, slug in (("4  &", "split4"), ("16 &", "split16"), ("24 &", "split24")):
        cells = _table_row(tex, sweep_hdr, row_label)
        est = _meta_of(f"validation-study-{slug}.json")["estimated_usd"]
        out.append((f"tab:splitsweep {slug} priced bound", cells[4][0], 0.005, est))

    # tab:results: all ten pre-registered runs, retained as the record of what the defect
    # invalidated. The paper says no claim rests on its Delta column, which is true and is not a
    # licence for the numbers to drift. Tolerance is 0.006 because this table prints to two
    # decimals in most cells (0.975 appears as 0.97), unlike the three-decimal tables above.
    results_hdr = r"Run & Regime & Seeds & Seed & B (prompt-only)"
    for row_label, slug in (
        ("1 &", ""),
        ("2 &", "-multi"),
        ("3 &", "-multihard"),
        ("4 &", "-feedback"),
        ("5 &", "-plan"),
        ("6 &", "-crossfamily"),
        ("7 &", "-oracle"),
        ("8 &", "-fixbudget"),
        ("9 &", "-fbactlo"),
        ("10 &", "-fbacthi"),
    ):
        cells = _table_row(tex, results_hdr, row_label)
        runs = _runs(f"validation-study{slug}.json")
        n = row_label.strip(" &")
        b = mean(r["B"]["opt_heldout"] for r in runs)
        d = mean(r["D"]["opt_heldout"] for r in runs)
        out.append((f"tab:results run{n} seeds", cells[2][0], 0, len(runs)))
        out.append(
            (
                f"tab:results run{n} seed",
                cells[3][0],
                0.006,
                mean(r["A"]["seed_heldout"] for r in runs),
            )
        )
        out.append((f"tab:results run{n} B", cells[4][0], 0.006, b))
        out.append((f"tab:results run{n} D", cells[5][0], 0.006, d))
        out.append((f"tab:results run{n} delta", cells[6][0], 0.006, d - b))

    # tab:residual, all 24 cells, replayed from the cassettes through the production loop.
    residual_hdr = r"Constraint (final-draft satisfaction) & run 3"
    rates = _residual_rates()
    residual_cols = ("run3", "run8", "run10")
    for iid in (
        "exact_word_count",
        "word_max_length",
        "forbidden_word",
        "num_bullets",
        "end_with",
        "include_keyword",
        "all_lowercase",
        "no_comma",
    ):
        label = "\\texttt{" + iid.replace("_", "\\_") + "}"
        cells = _table_row(tex, residual_hdr, label)
        for col, run_key in enumerate(residual_cols, start=1):
            want = rates.get(run_key, {}).get(iid)
            if want is None:
                continue
            out.append((f"tab:residual {iid} {run_key}", cells[col][0], 0.005, want))

    # tab:c1's run-3 row and the arm-D mechanism table, both cassette-derived confusion
    # matrices. The mechanism table was the least-checked and most load-bearing table in the
    # paper before the second audit; its cells are now read positionally rather than via prose.
    c1_hdr = r"Critique gene & $n$ & base rate & tp & fp & fn & PASS rate"
    st3 = _detector_stats("study-multihard.json", [0, 1, 2, 3, 4], {})
    c1_cells = _table_row(tex, c1_hdr, "run 3, intrinsic")
    for col, key in ((1, "n"), (2, "base_rate"), (3, "tp"), (4, "fp"), (5, "fn")):
        out.append((f"tab:c1 run3 {key}", c1_cells[col][0], 0.005, st3[key]))
    # PASS rate is the share of drafts the critique accepted: (tp + fp) / n.
    out.append(
        (
            "tab:c1 run3 PASS rate",
            c1_cells[6][0],
            0.005,
            (st3["tp"] + st3["fp"]) / st3["n"],
        )
    )

    mech_hdr = r"critique & PASS rate & tp & fp & fn & tn"
    mech_rows = _evolved_conditions()
    for seed in (0, 1, 2):
        for kind, label in (
            ("seed", f"seed rubric, seed {seed}"),
            ("evolved", f"\\textbf{{evolved}}, seed {seed}"),
        ):
            cond = next(c for c in mech_rows if c.seed == seed and c.kind == kind)
            cells = _table_row(tex, mech_hdr, label)
            out.append((f"mech s{seed} {kind} PASS", cells[1][0], 0.005, cond.pass_rate))
            for col, key in ((2, "tp"), (3, "fp"), (4, "fn"), (5, "tn")):
                out.append((f"mech s{seed} {kind} {key}", cells[col][0], 0, cond.stats[key]))

    # tab:ciexcl: the interval-correction exhibit. Its per-seed column and its CORRECTED column
    # are both recomputed; its "as published" column deliberately is not, because those numbers
    # are the output of the percentile-index bug that this table documents. They are historical
    # artifacts of a fixed defect, so no current code can reproduce them -- see the exemption
    # list in tests/test_paper_table_coverage.py.
    ciexcl_hdr = r"Result & per-seed & as published & corrected"
    for row_label, kind, has_corrected in (
        ("Actuator, powered", "actuator", True),
        ("Actionability, hi", "actionability", True),
        ("Null, run", "null3", False),  # the paper prints "unchanged" rather than an interval
    ):
        cells = _table_row(tex, ciexcl_hdr, row_label)
        per_seed = _per_seed_deltas(kind)
        for i, want in enumerate(per_seed):
            out.append((f"tab:ciexcl {kind} seed{i}", cells[1][i], 0.0005, want))
        if has_corrected:
            lo, hi = _corrected_ci(kind)
            out.append((f"tab:ciexcl {kind} corrected lo", cells[3][0], 0.0005, lo))
            out.append((f"tab:ciexcl {kind} corrected hi", cells[3][1], 0.0005, hi))

    # --- D7: the 135 replicate re-scorings (sec:repeat) --------------------------------
    # Every cell of tab:repeat and tab:sigmagenome, recomputed from the replicate outputs. These
    # supersede the offline noise estimate wherever they overlap, so drift between the two would
    # leave the paper asserting a superseded number as current.
    reps = _repeat_results()
    if reps:
        repeat_hdr = r"level & single draw & $\dd$ ($K{=}5$) & sd(draw)"
        for lvl in ("L1", "L2", "L3"):
            r = reps.get(lvl)
            if r is None:
                continue
            cells = _table_row(tex, repeat_hdr, f"${lvl[0]}_{lvl[1]}$ &")
            out.append((f"tab:repeat {lvl} K", 5.0, 0, len(r.replicate_deltas)))
            out.append((f"tab:repeat {lvl} delta", cells[2][0], 0.0005, r.delta))
            out.append((f"tab:repeat {lvl} sd", cells[3][0], 0.00005, r.sd_between_replicates))
            out.append((f"tab:repeat {lvl} se", cells[4][0], 0.00005, r.standard_error))
            out.append((f"tab:repeat {lvl} t", cells[5][0], 0.005, r.t_stat()))

        mod = _repeat()

        # The critical values the verdicts use. Scraped from the caption that names them, so a
        # future edit cannot quietly swap in the normal 1.96 that would flip two verdicts.
        out.append(
            (
                "tex: t crit 10 df4",
                _scrape(tex, r"critical values are \$([\d.]+)\$ \(\$p<0.10\$\)"),
                0.0005,
                mod.T_CRIT_10[4],
            )
        )
        out.append(
            (
                "tex: t crit 05 df4",
                _scrape(tex, r"and \$([\d.]+)\$ \(\$p<0.05\$\)"),
                0.0005,
                mod.T_CRIT_05[4],
            )
        )

        sig = _repeat_sigma()
        if sig:
            sigma_hdr = r"level & prompt-only (B) & loop-level (D) & unoptimized (seed)"
            for lvl in ("L1", "L2", "L3"):
                cells = _table_row(tex, sigma_hdr, f"${lvl[0]}_{lvl[1]}$ &")
                for col, arm in ((1, "B"), (2, "D"), (3, "SEED")):
                    out.append(
                        (f"tab:sigmagenome {lvl} {arm}", cells[col][0], 0.00005, sig[(lvl, arm)])
                    )
            lo, hi = min(sig.values()), max(sig.values())
            out.append(
                (
                    "tex: sigma genome range factor",
                    _scrape(tex, r"a factor of \$([\d.]+)\$\. Two mechanisms"),
                    0.05,
                    hi / lo,
                )
            )
            out.append(
                ("tex: sigma genome lo", _scrape(tex, r"range is \$([\d.]+)\$ to"), 0.00005, lo)
            )
            out.append(
                ("tex: sigma genome hi", _scrape(tex, r"to \$([\d.]+)\$,\na factor"), 0.00005, hi)
            )
            out.append(
                (
                    "tex: seed control sd",
                    _scrape(tex, r"moves\nacross replicates with sd \$([\d.]+)\$"),
                    0.00005,
                    _repeat().seed_replicate_spread(5)[1],
                )
            )

    # tab:budgetsweep: the pre-registered budget axis. Checkable before the runs exist because
    # every column is derived -- cycles and effective depth are arithmetic on (mmc, cap, n_train),
    # and the priced bound comes from the SAME cost model the pre-spend guard consults, so the
    # paper cannot advertise a bound the guard would not enforce.
    budget_hdr = r"\texttt{mmc} & cap & promotable cycles & effective depth"
    for mmc, cap in BUDGET_SWEEP:
        cells = _table_row(tex, budget_hdr, f"{mmc}  &" if mmc < 100 else f"{mmc} &")
        cell = _budget_cell(mmc, cap)
        out.append((f"tab:budgetsweep mmc{mmc} value", cells[0][0], 0, mmc))
        out.append((f"tab:budgetsweep mmc{mmc} cap", cells[1][0], 0, cap))
        out.append((f"tab:budgetsweep mmc{mmc} cycles", cells[2][0], 0.005, cell["cycles"]))
        out.append((f"tab:budgetsweep mmc{mmc} depth", cells[3][0], 0.005, cell["depth"]))
        if cells[4]:  # the free row prints "(run 3b, free)" rather than a number
            out.append((f"tab:budgetsweep mmc{mmc} bound", cells[4][0], 0.005, cell["usd"]))

    # --- The split-size sweep, bought (sec:splitresult) --------------------------------
    # Every number in tab:splitresults, cell by cell, plus the prose that interprets it. The
    # table is the whole result of a $6.98 purchase, so an unchecked digit here is the most
    # expensive kind of typo available in this paper.
    noise = _noise_estimate()
    for row_label, slug in (
        ("4  &", "split4"),
        ("8  &", "c2b"),
        ("16 &", "split16"),
        ("24 &", "split24"),
    ):
        cells = _table_row(tex, r"seed-genome & $\dd(D-B)$", row_label)
        r = _split_row(slug)
        n = row_label.rstrip(" &")
        out.append((f"tex: split n={n} mmc", cells[1][0], 0, r.mmc))
        out.append((f"tex: split n={n} seed mean", cells[2][0], 0.00005, r.seed_mean))
        out.append((f"tex: split n={n} delta", cells[3][0], 0.0005, r.delta))
        for i, per_seed in enumerate(cells[4]):
            out.append((f"tex: split n={n} seed{i} delta", per_seed, 0.0005, r.per_seed[i]))
        out.append(
            (
                f"tex: split n={n} in sigma",
                cells[5][0],
                0.005,
                abs(r.delta) / noise.sigma_delta_mean,
            )
        )
        out.append((f"tex: split n={n} arm B accepted", cells[6][0], 0, r.b_accepted))
        out.append((f"tex: split n={n} arm B of", cells[6][1], 0, r.n_seeds))

    # Prediction (i)'s resolution: PASS-rate-zero genes, per size and pooled over the sweep.
    pass_zero = _sweep_pass_zero()
    for label, pattern, artifact in (
        (
            "tex: split PASS zero at 16",
            r"\\texttt\{PASS\} rate is \$0.00\$ on \$(\d)/3\$ arm-D genes",
            pass_zero["split16"],
        ),
        (
            "tex: split PASS zero at 24",
            r"\{=\}16\$ and \$(\d)/3\$ at",
            pass_zero["split24"],
        ),
        (
            "tex: split PASS zero pooled",
            r"Pooling all four sizes it is \$(\d+)/12\$",
            sum(pass_zero.values()),
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0, artifact))

    # The confound, quantified: promotable accept cycles and minibatch share. Recomputed from
    # each run's own recorded (mmc, n_train), so this checks the paper against what ran.
    cycles = [_split_row(g).cycles for _, g in _SPLIT_ROWS]
    prose_cycles = re.search(
        r"across the sweep as \$([\d.]+),\\,([\d.]+),\\,([\d.]+),\\,([\d.]+)\$", tex, re.DOTALL
    )
    for i, want in enumerate(cycles):
        got = float(prose_cycles.group(i + 1)) if prose_cycles else float("nan")
        out.append((f"tex: split promotable cycles {i}", got, 0.005, want))

    # The level-to-level swing, which is what survives the withdrawal of the L1 loss. Checked
    # because it is now the p0 section's load-bearing number: if sigma moves, this must too.
    _l1, _l2 = _decomposition(0)["delta"], _decomposition(1)["delta"]
    out.append(
        (
            "tex: p0 swing in sigma",
            _scrape(tex, r"\\approx\\; ([\d.]+) ,\n\\]"),
            0.05,
            (_l2 - _l1) / (noise.sigma_delta_mean * 2**0.5),
        )
    )
    out.append(
        (
            "tex: p0 swing magnitude",
            _scrape(tex, r"a swing of \$([\d.]+)\$ in \$\\dd\$"),
            0.0005,
            _l2 - _l1,
        )
    )

    # --- The seed-baseline curve, in units of the noise floor (sec:threats) -----------
    # Added after the section misreported one of these gaps by pooling run 3's five seeds
    # against three-seed runs. Every gap the prose states is now recomputed from the runs.
    out.append(
        (
            "tex: curve shared first-draft calls",
            _scrape(tex, r"first-draft prompts on \$(\d+)\$ calls each"),
            0,
            _shared_first_draft_calls(),
        )
    )
    out.append(
        (
            "tex: curve run3 three-seed mean",
            _scrape(tex, r"\(\$([\d.]+)\$ rather than the"),
            0.00005,
            _curve_seed_mean("run3"),
        )
    )
    out.append(
        (
            "tex: curve run3 five-seed mean",
            _scrape(tex, r"five-seed \$([\d.]+)\$ of Table"),
            0.00005,
            _curve_seed_mean("run3", shared_seeds_only=False),
        )
    )
    for label, pattern, pair in (
        (
            "tex: curve tool lift",
            r"\(run~4 over run~3\) is \$\+([\d.]+)\$ at \$([\d.]+)\\sigma\$",
            ("run3", "run4"),
        ),
        (
            "tex: curve actionability",
            r"\(run~10 over\s+run~9\) is \$\+([\d.]+)\$ at \$([\d.]+)\\sigma\$",
            ("run9", "run10"),
        ),
        (
            "tex: curve oracle vs tool-fed",
            r"it is clear:\s+\$-([\d.]+)\$ at \$([\d.]+)\\sigma\$",
            ("run4", "run7"),
        ),
        (
            "tex: curve more retries",
            r"more refine attempts\n?\s*\(\$\+([\d.]+)\$, \$([\d.]+)\\sigma\$\)",
            ("run7", "run8"),
        ),
        (
            "tex: curve oracle vs intrinsic",
            r"it is not: \$-([\d.]+)\$ at \$([\d.]+)\\sigma\$",
            ("run3", "run7"),
        ),
    ):
        gap, in_sigma = _curve_gap(*pair)
        m = re.search(pattern, tex, re.DOTALL | re.MULTILINE)
        out.append((label, float(m.group(1)) if m else float("nan"), 0.0005, abs(gap)))
        out.append((f"{label} sigma", float(m.group(2)) if m else float("nan"), 0.005, in_sigma))

    # The one gap in the curve the paper explicitly declines to claim. Sigma only: the caption
    # prints the two endpoints rather than their difference, so there is no gap value to scrape.
    out.append(
        (
            "tex: curve run10 vs tool sigma",
            _scrape(tex, r"that particular gap is\n\$([\d.]+)\\sigma\$"),
            0.005,
            _curve_gap("run4", "run10")[1],
        )
    )

    # The code-actionability effect, which the paper now reports WITHOUT an interval because
    # its seeds are partitions. The overlap it cites is checked, and so is the claim that the
    # seeds are not independent -- if that changed, an interval would become reportable again.
    out.append(
        (
            "tex: bcb actionability overlap",
            _scrape(tex, r"held-out overlap is \$([\d.]+)\\%\$"),
            0.05,
            _bcb_actionability_overlap() * 100,
        )
    )

    # Evidence 9's census. The paper distinguishes two shapes of the same flag and the counts
    # are the whole argument: a one-sided contrast (one arm starved, the other searched) is the
    # new defect, while both-arms-starved is the original retraction re-derived from scores.
    one_sided, no_contrast = _contrast_flag_census()
    for label, pattern, artifact in (
        ("tex: flagged runs total", r"(\w+) released runs trip", one_sided + no_contrast),
        (
            "tex: flagged both-arms-starved",
            r"In the other\s+(\w+) \\emph\{both\} arms",
            no_contrast,
        ),
    ):
        out.append((label, _scrape_word(tex, pattern), 0, artifact))

    # Between-cell spread of Δ, the sentence that licenses "one quantity plus noise".
    out.append(
        (
            "tex: split between-cell spread",
            _scrape(tex, r"spread of \$\\dd\$ \(\$([\d.]+)\$\)"),
            0.0005,
            pstdev([_split_row(g).delta for _, g in _SPLIT_ROWS]),
        )
    )

    # --- The measurement-noise floor (sec:threats) -------------------------------------
    # The component every interval in the study omits. Scraped from the prose that now
    # reports it, checked against the estimator, so a reworded sentence breaks loudly.
    for label, pattern, tol, artifact in (
        ("tex: noise sigma single", r"under one draw is \$([\d.]+)\$", 0.0005, noise.sigma_single),
        (
            "tex: noise sigma delta task",
            r"so \$\\sigma\(\\dd\)\$ on one task is \$([\d.]+)\$",
            0.0005,
            noise.sigma_delta_task,
        ),
        (
            "tex: noise sigma delta mean",
            r"0\.153/\\sqrt\{30\} \\;=\\; ([\d.]+)",
            0.00005,
            noise.sigma_delta_mean,
        ),
        (
            "tex: noise mean signed",
            r"between replicates, \$(-[\d.]+)\$, is confounded",
            0.0005,
            noise.mean_signed,
        ),
    ):
        out.append((label, _scrape(tex, pattern), tol, artifact))

    # The five replicate means, in the order the prose lists them.
    prose_means = re.search(
        r"Their means are \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$",
        tex,
        re.DOTALL,
    )
    want_means = [noise.replicate_means[k] for k in _NOISE_REPLICATE_ORDER]
    for i, want in enumerate(want_means):
        got = float(prose_means.group(i + 1)) if prose_means else float("nan")
        out.append((f"tex: noise replicate mean {i}", got, 0.00005, want))

    # The headline effects restated in sigma units, in both the abstract and sec:threats.
    for label, pattern, delta in (
        ("tex: sigma units L1", r"\$L_1\$ loss\s+\$-0.042\$ at \$([\d.]+)\\sigma\$", -0.0417),
        ("tex: sigma units L2", r"\$\+0.060\$ at \$([\d.]+)\\sigma\$", 0.0600),
        ("tex: sigma units L3", r"\$-0.002\$ at \$([\d.]+)\\sigma\$", -0.0017),
        ("tex: sigma units act", r"\$\+0.063\$ at \$([\d.]+)\\sigma\$", 0.0630),
        ("tex: abstract L1 sigma", r"puts that \$-0.042\$ at \$([\d.]+)\\sigma\$", -0.0417),
    ):
        out.append((label, _scrape(tex, pattern), 0.005, noise.in_sigma(delta)))

    # The reproducibility registry's composition, which drifted once already: the paper said
    # "seventeen ... the eleven below plus the two hand-written conditions", which is 13.
    # These counts are spelled out in the prose, so they need the word scraper.
    superseded, replayable = _registry_counts()
    for label, pattern, artifact in (
        ("tex: registry total", r"registers (\w+) committed runs", superseded + replayable),
        ("tex: registry superseded", r"(\w+) are\s+labelled \\textbf\{superseded\}", superseded),
        ("tex: registry replayable", r"The remaining (\w+) were recorded", replayable),
        ("tex: registry verifies N", r"verifies (\w+) runs rather than", replayable),
        ("tex: retraction superseded count", r"script registers (\w+) superseded runs", superseded),
    ):
        out.append((label, _scrape_word(tex, pattern), 0, artifact))

    # The two reflection models, asserted as a set rather than a number: the check is that the
    # paper names exactly the model each half of the study used, in the right half.
    corrected_models, pre_models = _reflection_models()
    for label, want, marker in (
        ("tex: corrected-harness reflection model", corrected_models, "on every run recorded"),
        ("tex: pre-correction reflection model", pre_models, "which is the\nCLI default"),
    ):
        if len(want) != 1:
            raise ValueError(
                f"{label}: the released runs use {sorted(want)} reflection models, so the "
                "paper cannot name one; widen the sentence rather than the tolerance"
            )
        named = want.pop().split("/")[-1].replace("-", r"\-")
        found = 1.0 if re.search(rf"\\texttt\{{{named}\}}[^.]*?{marker}", tex, re.DOTALL) else 0.0
        out.append((label, found, 0, 1.0))

    # How many contrast runs the correction superseded, and how many of their deltas really
    # were zero. The paper said "eleven" for a set of fifteen, and called all of their zeros
    # identities when four are not zero at all.
    n_pre, n_zero, max_abs = _precorrection_counts()
    for label, pattern, artifact in (
        (
            "tex: pre-correction contrast runs",
            r"discarding every contrast run we had bought before it ---\s+(\w+) of them",
            n_pre,
        ),
        ("tex: intro pre-correction runs", r"Our first (\w+) contrast runs measured", n_pre),
        (
            "tex: defect appendix pre-correction runs",
            r"Our first (\w+) contrast runs reported",
            n_pre,
        ),
        (
            "tex: defect appendix exact zeros",
            r"was exactly zero (\w+) times",
            n_zero,
        ),
        (
            "tex: superseded appendix run count",
            r"Of the\s+(\w+) contrast runs recorded under the pre-correction harness",
            n_pre,
        ),
        (
            "tex: superseded appendix exact zeros",
            r"harness, (\w+) have a mean \\dd\\ of\s+exactly zero",
            n_zero,
        ),
        ("tex: conclusion pre-correction runs", r"\\textbf\{(\w+)\} of them, across", n_pre),
        (
            "tex: conclusion exact zeros",
            r"(\w+) reported \$\\dd\(D\{-\}B\)\$ of exactly zero",
            n_zero,
        ),
    ):
        out.append((label, _scrape_word(tex, pattern), 0, artifact))
    out.append(
        (
            "tex: largest pre-correction |mean delta|",
            _scrape(tex, r"the largest being the powered actuator run at\s+\$-([\d.]+)\$"),
            0.0005,
            max_abs,
        )
    )

    return out


def main() -> int:  # pragma: no cover - CLI convenience
    failures = []
    checks = _checks()
    for label, claimed, tol, actual in checks:
        ok = abs(actual - claimed) <= tol
        if not ok:
            failures.append((label, claimed, actual))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:32} paper={claimed:<9} artifact={actual}")
    print(
        f"\n{len(checks) - len(failures)}/{len(checks)} paper claims match the released artifacts."
    )
    if failures:
        print("\nDRIFT:")
        for label, claimed, actual in failures:
            print(f"  {label}: paper says {claimed}, artifacts say {actual}")
        return 1
    print(
        "\nNOT machine-checkable here (verified by hand, listed so the gap is visible):\n"
        "  - Evidence 1's per-arm gene-change counts: they replay only against the pre-fix\n"
        "    commit, since the corrected reflective dataset changes every cassette key.\n"
        "    Re-verified 2026-07-28 at fe670a8: generate 5/15, ensure_correct 0/15, and\n"
        "    per arm B 3/5, C 0/5, D 2/5 generate with 0/5 ensure_correct.\n"
        "  - Evidence 7's 70 swallowed misses and the 0.666 -> 0.681 substitution, which\n"
        "    require a deliberately mismatched replay.\n"
        "  - The stopped mmc=160 run's partial spend, whose cassette is not released.\n"
        "  - Every cited external result, and every prose claim that is not one of the\n"
        "    `tex:` numbers above -- those are scraped from main.tex, the rest are not."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
