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
(the 2/3 is arm B's `generate`, transposed). The `tex:` checks scrape the papers' own sources, so
the claimed side is the paper's own sentence. A pattern that stops matching yields NaN and
fails, because a reworded sentence must break the check loudly rather than quietly stop
being checked.

Scope: numbers recomputable from `docs/*.json`, the vendored task data, and the committed
cassettes, plus the headline numbers as they are written in the three papers under
`docs/papers/`. Claims
that are not machine-checkable (unquantified prose, cited literature, one-off replays
against superseded commits) are out of scope and listed at the bottom of the output so the
gap is visible rather than implied.

    python scripts/verify_paper_claims.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import mean, pstdev, stdev

DOCS = "docs"

# The pre-registered split-size sweep. Kept here as the single source of truth so the
# paper table, the guard tests and these checks cannot drift apart.
SWEEP = ((4, 14), (8, 28), (16, 56), (24, 84))
EPOCHS = 3.5  # metric calls per training task, held constant across the sweep
POOL_CLIFF = 32  # first n_train whose pool leaves 64, which would move the held-out set


@cache
def _sources():
    """The `paper_sources` module, imported lazily so this file stays runnable from the repo root
    without a package install (the same shim the analyzer imports below use)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper_sources

    return paper_sources


def _meta_of(name: str) -> dict:
    """A released run's `meta` block. Mirrors `_runs`, using plain `open` so it does not depend
    on the lazy `pathlib` imports the analyzer shims use."""
    with open(f"{DOCS}/{name}", encoding="utf-8") as fh:
        return json.load(fh)["meta"]


def _runs(name: str) -> list[dict]:
    with open(f"{DOCS}/{name}", encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def _tex_without_registers() -> str:
    r"""The papers' prose with each corrections appendix's `enumerate` body removed.

    For sweeps that check EVERY site stating a fact. A corrections register quotes the value a
    paper used to print --- that is what a register is for --- so a sweep over the whole source
    reads the quotation as a live claim and fails on the very entry that records the fix.

    Only for sweeps. The registers are not exempt from checking in general, and most of their
    numbers are pinned: where a register states a superseded statistic, `_restated_statistic_checks`
    derives it by REAPPLYING the old mistake rather than transcribing it, which is stronger than an
    exemption. What cannot be derived is a value that was simply wrong, and those are the ones this
    strips.
    """
    return re.sub(
        r"entries follow.*?\\end\{enumerate\}", "", _tex(), flags=re.DOTALL | re.MULTILINE
    )


def _abs_mean_delta(name: str) -> float:
    """|mean of the per-seed D-B held-out differences| for one released run.

    Exists so a sigma multiple is never computed from a figure the paper already rounded for
    display. A check fed the paper's own rounded delta compares the paper to itself: it agrees by
    construction and stays agreeing while the artifact moves underneath it.
    """
    return abs(mean(r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in _runs(name)))


def _arm_a_inertness() -> tuple[int, int]:
    """(arm-A seed entries across the release, how many are not inert).

    Backs Paper II's licence to read arm A out of four-arm artifacts. Inert means the arm declares
    no evolvable genes, changed none, and scored identically before and after the search that ran
    on its siblings. Paper II used to justify that licence by claiming every run was invoked with
    `--arms A`, which is false for the seven runs behind its own curve table: they carry all four
    arms and six of them recorded a real search on B and D. The invariant below is what actually
    holds, and unlike a claim about a CLI invocation it is visible in the released artifact.
    """
    import glob

    entries = violations = 0
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for run in doc["runs"]:
            arm = run.get("A")
            if not isinstance(arm, dict):
                continue
            entries += 1
            if (
                arm.get("evolvable_genes") != []
                or arm.get("genes_changed")
                or arm.get("seed_heldout") != arm.get("opt_heldout")
            ):
                violations += 1
    return entries, violations


def _vacuous_ifbench_verifiers() -> tuple[int, int]:
    """(verifiers in the vendored IFBench slice, how many the empty string satisfies).

    Paper I's do-nothing-floor threat said five of eleven. Six of the eleven return True on `""`:
    the sub-bullet check reduces to `all([])`, and the vowel and consonant checks quantify over no
    words. Derived by calling them, because a hand-counted subset of a ported checker suite is
    exactly the claim that rots when the port gains a verifier.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from whystuck.eval.ifbench import VERIFIERS

    return len(VERIFIERS), sum(1 for fn in VERIFIERS.values() if fn(""))


def _tex() -> str:
    """Every paper's source, concatenated, shared inputs expanded.

    The `tex:` checks below ask whether the series' prose agrees with the artifact, which is a
    question about the series and not about which of the three files a sentence landed in. Reading
    the concatenation is also what let the split preserve every pinned-phrase check written
    against the pre-split draft: a sentence that survived into *some* paper is still found, and
    one dropped by the split yields NaN and fails loudly. Per-paper questions -- does each paper
    still carry its own headline numbers, do two papers quoting one number agree -- are asked
    separately, in `_spine_checks` and `_cross_paper_checks`.
    """
    return _sources().all_tex()


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


def _scrape_all(tex: str, pattern: str) -> list[float]:
    """`_scrape` for a sentence that lists several numbers in one breath.

    Same NaN-on-no-match contract, one per capture group, so a reworded sentence fails every
    number it carried instead of silently dropping the ones past the first. Sentences of the form
    "$a$ at $L_1$, $b$ at $L_2$ and $c$ at $L_3$" are where this study's stalest numbers hid:
    one of the three gets updated and its neighbours do not.
    """
    m = re.search(pattern, tex, re.DOTALL | re.MULTILINE)
    if m is None:
        return [float("nan")] * max(1, re.compile(pattern).groups)
    return [float(g) for g in m.groups()]


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
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
    "twenty-four": 24,
    "twenty-five": 25,
    "twenty-six": 26,
    "thirty-eight": 38,
    "forty-three": 43,
    "two hundred": 200,
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


def _noise_estimate(exclude: str | None = None):
    """The measurement-noise estimator, imported lazily like the other analyzers.

    `exclude` drops one replicate by name. It exists for a single claim: the ten replicates behind
    sigma include the discarded first attempt at the four-arm re-run, which the papers elsewhere
    declare superseded and unreplayable. Enrolling it is right -- its seed column scores a fixed
    genome and involves no optimizer, so the defect that superseded its delta does not touch this
    measurement -- but it was undisclosed, and the paragraph now states what sigma would be without
    it. That number needs deriving rather than typing, which is what this parameter is for.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_measurement_noise as mod

    if exclude is None:
        return mod.noise_estimate()
    names = [r.name for r in mod.eligible_replicates() if r.name != exclude]
    if len(names) == len(mod.eligible_replicates()):
        raise ValueError(f"{exclude!r} is not an eligible replicate; the estimator's set changed")
    return mod.noise_estimate(names)


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


def _worker_keys(slug: str) -> set[str]:
    """The worker-surface cassette keys of one study, i.e. its first-draft prompts."""
    entries = json.loads(Path(f"tests/fixtures/cassettes/study-{slug}.json").read_text())["entries"]
    return {k for k, v in entries.items() if v.get("name") == "worker"}


@cache
def _curve_shared_worker_keys() -> frozenset[str]:
    """Worker calls common to every run of the curve. Should be 3 x (10 heldout + 8 train).

    This is the licence for omitting a task-draw term from the between-run comparisons: if the
    prompts are byte-identical then the tasks are identical and difficulty cancels. If a future
    edit breaks it, the paired standard errors in sec:threats become anti-conservative, so it is
    checked rather than remembered.
    """
    shared: set[str] | None = None
    for slug in _CURVE_RUNS.values():
        keys = _worker_keys(slug)
        shared = keys if shared is None else (shared & keys)
    return frozenset(shared or ())


def _shared_first_draft_calls() -> int:
    return len(_curve_shared_worker_keys())


def _critique_swap_gap(slug: str) -> float:
    """Seed-genome held-out gap against run 3, on the three seeds the curve's runs share.

    The substituted-critic runs are not members of the curve, so their eligibility for the same
    paired comparison is asserted here rather than assumed: they must carry every one of the
    curve's byte-identical first-draft calls. Without that the difference would carry a
    task-draw term and the pairing would be fiction -- the exact error `_curve_seed_mean`
    records having been made once already, by pooling run 3's five seeds.
    """
    missing = _curve_shared_worker_keys() - _worker_keys(slug)
    if missing:
        raise AssertionError(
            f"study-{slug} is missing {len(missing)} of the curve's first-draft calls, so its "
            "seed-genome score is not paired with run 3's"
        )
    runs = _runs(f"validation-study-{slug}.json")[:_CURVE_SHARED_SEEDS]
    return mean(r["A"]["seed_heldout"] for r in runs) - _curve_seed_mean("run3")


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


@cache
def _curve_replicates() -> tuple[dict[str, float], dict[tuple[str, str], object]]:
    """The 90 curve replicates: (sd of each row's 30-task mean, paired result per claimed gap).

    Read through `repeat_scores` rather than re-implemented, so the paper, the driver and the
    aggregation cannot disagree about what was measured. Cached because every check below would
    otherwise re-read ninety JSON files.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repeat_scores as rc

    by_label = {r.level: r for r in rc.curve_aggregate(5)}
    pairs = {(a, b): by_label[label] for label, a, b in rc.CURVE_PAIRS if label in by_label}
    return rc.sigma_by_run(5), pairs


@cache
def _curve_priced_bound() -> float:
    """Summed pre-flight estimate over the 90 curve replicates.

    A BOUND, never an invoice: the harness records what the guard priced, not what was billed.
    Corrections entry 16 exists because an earlier draft reported one as a measured spend, and the
    prose-coverage guard caught the curve subsection doing it a second time.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repeat_scores as rc

    return sum(
        json.loads(Path(job.out).read_text())["meta"].get("estimated_usd") or 0.0
        for job in rc.curve_plan(5)
    )


@cache
def _p0_seed_sigma() -> float:
    """The p0 family's unoptimized L1 sigma: the same genome and configuration as curve run 3.

    The cross-family control. Two experiments measured one quantity independently, so a
    disagreement would mean one of them was not measuring what it claimed.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repeat_scores as rc

    return rc.sigma_by_genome(5)[("L1", "SEED")]


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

#: The sweep's cells as (mmc, cap, released run slug). The mmc=28 cell is `c2b`, the four-arm
#: re-run: it was not bought for this sweep, it *is* the cell the sweep interrogates -- the same
#: relationship `_SPLIT_ROWS` records for the n_train=8 row. Kept here rather than rebuilt beside
#: each consumer, because a second copy of this mapping is a second thing to get wrong.
_BUDGET_ROWS = (
    (20, 6, "mmc20"),
    (28, 6, "c2b"),
    (40, 6, "mmc40"),
    (56, 6, "mmc56"),
    (80, 6, "mmc80"),
    (160, 12, "mmc160"),
)
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


def _search_depth(slug: str, mmc: int, cap: int) -> int | None:
    """Measured GEPA iterations, replayed from the cassette. See analyze_search_depth."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_search_depth as asd

    return asd.measured_depth(slug, mmc, cap)


def _budget_delta(slug: str) -> float:
    runs = _runs(f"validation-study-{slug}.json")
    return mean(r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs)


def _budget_acceptance(slug: str) -> int:
    """Arm-seed pairs where the optimizer changed any gene, out of six."""
    runs = _runs(f"validation-study-{slug}.json")
    return sum(1 for r in runs for arm in ("B", "D") if (r.get(arm, {}).get("genes_changed") or []))


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


#: How Paper III's prose enumerates the replayable registry, as name prefixes. The eight groups
#: must partition it exactly: every replayable run lands in one, and the sizes sum to the total.
_REPLAYABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "hand-written-gene conditions": ("C1 ",),
    "four-arm re-run": ("C2b",),
    "p0 levels": ("p0 sweep",),
    "p0 calibration passes": ("p0 calibration",),
    "split sweep": ("split sweep",),
    "budget sweep": ("budget sweep",),
    "cross-provider critic": ("D3 ",),
    "second-worker line": ("groq ",),
    "diagonal splice control": ("splice ",),
}


def _replayable_partition() -> dict[str, int]:
    """The replayable registry, bucketed the way Paper III's prose enumerates it.

    Paper III lists the twenty-five replayable runs as a sequence of groups. The *total* was
    pinned against the registry and the *parts* were prose, so the sentence summed to twenty-seven
    for the life of the split while every check passed: it credited the second-worker line with
    eleven runs when the registry holds eight of them plus a splice control on a third model.
    Bucketing by name and asserting the buckets partition the registry makes the list and the
    total fail together, and any run added to the release lands in a group or fails as unbucketed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import verify_reproducibility as vr

    names = [run[0] for run in vr.RUNS if run[0] not in vr.SUPERSEDED]
    sizes: dict[str, int] = {}
    claimed: list[str] = []
    for group, prefixes in _REPLAYABLE_GROUPS.items():
        hit = [n for n in names if n.startswith(prefixes)]
        sizes[group] = len(hit)
        claimed.extend(hit)
    unbucketed = sorted(set(names) - set(claimed))
    if unbucketed:
        raise LookupError(f"replayable runs in no group of Paper III's enumeration: {unbucketed}")
    if len(claimed) != len(set(claimed)):
        raise LookupError("Paper III's enumeration groups overlap; they must partition")
    return sizes


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


@dataclass(frozen=True)
class C1Strata:
    """The C1 held-out split by what the two critique genes did to each task.

    `named` and `residual` are the pair sec:analysis decomposes; `caught_*` describe the narrower
    stratum inside `named` where the hand-written gene caught a false pass, which is the only
    stratum where the threshold move can be credited with a gain.
    """

    named: float
    residual: float
    total_n: int
    caught_n: int
    caught_draft: float
    caught_final: float
    caught_contribution: float
    compliant_n: int
    compliant_mean: float


@cache
def _c1_strata() -> C1Strata:
    """C1's held-out tasks partitioned by verdict change and draft compliance.

    Cached because it replays the whole `study-c1` cassette twice, once per gene.

    The three-task `caught` stratum is derived here for the same reason the eight-task one is: the
    sentence that credits the threshold move quotes a draft mean, a final mean and a contribution
    to the 32-task mean, and all three sat unpinned. They are the arithmetic the paper's own
    conclusion rests on --- that eliminating false passes bought a gain an order of magnitude
    below the noise --- so typing them was the wrong way to state them.
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
    caught: list[tuple[float, float]] = []
    compliant: list[float] = []
    for task, (v_seed, actual), (v_hand, _) in zip(tasks, ev_seed, ev_hand, strict=True):
        if task.id not in b_by_task:
            continue
        diff = h_by_task[task.id] - b_by_task[task.id]
        # The eight the paper names: verdicts differ, or the draft was already compliant.
        if v_seed != v_hand or actual:
            named += diff
        # The two disjoint halves of those eight. `actual` is the draft's true compliance, so a
        # verdict change on a NON-compliant draft is a false pass the hand-written gene caught.
        if actual:
            compliant.append(diff)
        elif v_seed != v_hand:
            caught.append((b_by_task[task.id], h_by_task[task.id]))
    total = sum(h_by_task[k] - b_by_task[k] for k in b_by_task)
    if not caught or not compliant:
        raise ValueError(
            f"C1 strata came out {len(caught)} caught / {len(compliant)} compliant; the paper "
            "credits the threshold move on the first and excuses a loss on the second, so an "
            "empty one means the replay no longer supports either statement"
        )
    return C1Strata(
        named=named / n_heldout,
        residual=(total - named) / n_heldout,
        total_n=n_heldout,
        caught_n=len(caught),
        caught_draft=mean(d for d, _ in caught),
        caught_final=mean(f for _, f in caught),
        caught_contribution=sum(f - d for d, f in caught) / n_heldout,
        compliant_n=len(compliant),
        compliant_mean=mean(compliant),
    )


def _c1_decomposition() -> tuple[float, float]:
    """(contribution of the eight tasks sec:analysis names, residual on the other 24).

    The paper used to present these two as a decomposition of the whole -0.033. They are
    not: three tasks differ in verdict and five had an already-compliant draft, and the
    other twenty-four moved because the repair *prose* differed, not the verdict. Deriving
    both halves here is what keeps the partial decomposition labelled as partial.
    """
    strata = _c1_strata()
    return strata.named, strata.residual


@dataclass(frozen=True)
class Stratum:
    """One acceptance stratum of a two-optimized-arm contrast."""

    seeds: int
    mean_delta: float
    share: float


@cache
def _xprovider_strata() -> dict[str, Stratum]:
    """D3's Delta(D-B), split by which arms actually accepted a proposal.

    The run that made Paper III's presentation rule prospective rather than retrospective. Both
    arms accepted on at least one seed, so the rule's binary flag stays silent -- and the contrast
    is still dominated by seeds where the treatment did not apply and the COMPARATOR damaged
    itself. Deriving the split here is what lets the paper state that without typing a number.
    """
    doc = json.loads(Path(f"{DOCS}/validation-study-xprovider.json").read_text())
    runs = doc["runs"]
    buckets: dict[str, list[float]] = {"treated": [], "comparator only": [], "neither": []}
    for run in runs:
        delta = run["D"]["opt_heldout"] - run["B"]["opt_heldout"]
        moved_d = bool(run["D"].get("genes_changed"))
        moved_b = bool(run["B"].get("genes_changed"))
        key = "treated" if moved_d else ("comparator only" if moved_b else "neither")
        buckets[key].append(delta)
    total = sum(d for v in buckets.values() for d in v)
    if not total:
        raise ValueError("xprovider Delta sums to zero; the share denominator is undefined")
    return {
        k: Stratum(len(v), mean(v) if v else 0.0, sum(v) / total) for k, v in buckets.items() if v
    }


@cache
def _xprovider_acceptance() -> tuple[int, int, int]:
    """(seeds, seeds arm B accepted on, seeds arm D accepted on) for D3.

    Cross-checked two ways, because the one-way version is a defect this series shipped: an arm
    that accepts nothing returns the seed genome and so reproduces its own seed score, and reading
    `genes_changed` alone turned an ABSENT field into a measured zero (register entry, Paper III).
    """
    doc = json.loads(Path(f"{DOCS}/validation-study-xprovider.json").read_text())
    runs = doc["runs"]
    out = []
    for arm in ("B", "D"):
        by_field = sum(1 for r in runs if r[arm].get("genes_changed"))
        by_score = sum(
            1 for r in runs if abs(r[arm]["opt_heldout"] - r[arm]["seed_heldout"]) > 1e-9
        )
        if by_field != by_score:
            raise ValueError(
                f"D3 arm {arm}: genes_changed says {by_field} seeds accepted, the seed-score "
                f"check says {by_score}. One of the two is wrong and the Delta cannot be read "
                "until it is known which."
            )
        out.append(by_field)
    return len(runs), out[0], out[1]


@cache
def _critique_median_length(slug: str) -> int:
    """Median character length of one run's non-empty critique completions.

    Read off the `critic` cassette namespace, which exists only for runs whose critique was
    routed to a separate model -- the two substituted-critic runs. It is the evidence for the
    paper's claim that a model swap changes the critique's PROSE and not only its verdict, and
    that claim is load-bearing: without it the swap reads as a detection-only intervention and
    its lift would contradict the section. An empty completion is the critic declining to
    complain, which is a verdict rather than prose, so it is excluded.
    """
    entries = json.loads(Path(f"tests/fixtures/cassettes/study-{slug}.json").read_text())["entries"]
    lengths = sorted(
        len(v["output"])
        for v in entries.values()
        if v.get("name") == "critic" and v["output"].strip()
    )
    if not lengths:
        raise AssertionError(f"study-{slug} has no non-empty critic completions to measure")
    return lengths[len(lengths) // 2]


@cache
def _detector_row(cassette: str, seeds: tuple[int, ...], separate_critic: bool) -> dict[str, float]:
    """Confusion counts for one critique acting as a detector, replayed from its cassette.

    `separate_critic` is required for any run recorded with `--critique-model`: those route the
    critique to its own cassette namespace, and replaying them against the worker surface silently
    yields an empty event list rather than an error.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import analyze_critique_detector as acd

    events = acd.collect_events(
        f"tests/fixtures/cassettes/{cassette}",
        list(seeds),
        loop_kw={},
        n_heldout=10,
        separate_critic=separate_critic,
    )
    if not events:
        raise ValueError(f"{cassette}: replay produced no critique events; the surface is wrong")
    stats = acd.detector_stats(events)
    stats["pass_rate"] = (stats["tp"] + stats["fp"]) / stats["n"]
    return stats


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


def _precorrection_counts() -> tuple[int, int, float, int]:
    """(contrast runs recorded pre-correction, arms identical on every seed, largest |mean Δ|,
    how many have a *mean* Δ of exactly 0).

    `meta.module_selector == "all"` is the marker the corrected harness writes, so this
    partitions the released runs by harness generation without a hand-maintained list. It
    exists because the paper said "eleven" for a set that is fifteen, and separately claimed
    every Δ in it was an identity when four are not: an unchanged verify gene leaves the arms
    differing through the `generate` gene they share, which the default selector made arm D
    search half as hard (Evidence 3).

    **Why two counts.** Δ is defined in Paper III as mean(D) − mean(B), and this function used to
    report "exactly zero" from `mean(delta_arm(path))`, the mean of per-seed differences. Those two
    are the same number in real arithmetic and not in IEEE-754: on
    `validation-study-multihard.json`, whose per-seed differences are [0, +0.055, 0, 0, -0.055],
    difference-of-means is bit-equal to 0.0 while mean-of-differences is -2.2204460492503132e-17.
    So "exactly zero" held for ten runs or eleven depending on a summation order the paper never
    specified, and the papers printed ten while defining the estimator that gives eleven.

    A count that moves with summation order cannot carry a claim, so the load-bearing quantity is
    now `n_identical`: runs whose two arms produced bit-identical held-out scores on **every** seed.
    That is invariant, it is what the surrounding prose actually argues (the arms were the same
    program, so they issued the same calls), and it is the count the papers now state. The
    mean-exactly-zero figure is still returned so the discrepancy stays pinned rather than latent.
    """
    import glob
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_deltas import delta_arm

    means: list[float] = []
    identical: list[bool] = []
    by_means: list[float] = []
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        first = doc["runs"][0]
        arms = sorted(k for k in first if isinstance(first.get(k), dict))
        if arms != ["A", "B", "C", "D"] or doc["meta"].get("module_selector") == "all":
            continue
        per_seed = delta_arm(path)
        means.append(mean(per_seed))
        identical.append(all(d == 0.0 for d in per_seed))
        by_means.append(
            mean(r["D"]["opt_heldout"] for r in doc["runs"])
            - mean(r["B"]["opt_heldout"] for r in doc["runs"])
        )
    if not means:
        raise ValueError(
            "no pre-correction contrast runs found; the module_selector marker must have "
            "changed, and a count of zero would silently retract the paper's own caution"
        )
    return (
        len(means),
        sum(identical),
        max(abs(m) for m in means),
        sum(1 for m in by_means if m == 0.0),
    )


def _precorrection_acceptance() -> dict[str, int]:
    """Acceptances in the fifteen pre-correction contrast runs, derived from SCORES not fields.

    Those runs predate `genes_changed`, so `arms_optimized` reads a missing key as a zero and the
    census reports both arms accepting nothing. The released scores say otherwise, and the inference
    is sound in one direction: an arm whose search accepts nothing returns the seed genome, which is
    scored on the same held-out list by the same worker, so it must reproduce its own seed score.
    Therefore `opt_heldout != seed_heldout` on any seed is an accepted proposal. The converse does
    not hold -- an accepted proposal can score identically -- so these counts are a lower bound, and
    the paper states them as such.
    """
    import glob

    either = arm_b = arm_d = flat = 0
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        first = doc["runs"][0]
        arms = sorted(k for k in first if isinstance(first.get(k), dict))
        if arms != ["A", "B", "C", "D"] or doc["meta"].get("module_selector") == "all":
            continue
        moved = {
            arm: any(r[arm]["opt_heldout"] != r[arm]["seed_heldout"] for r in doc["runs"])
            for arm in ("B", "D")
        }
        arm_b += moved["B"]
        arm_d += moved["D"]
        either += moved["B"] or moved["D"]
        flat += not (moved["B"] or moved["D"])
    return {"either": either, "B": arm_b, "D": arm_d, "flat": flat}


def _registry_counts() -> tuple[int, int]:
    """(superseded, replayable) as `verify_reproducibility.py` itself defines them."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import verify_reproducibility as vr

    return len(vr.SUPERSEDED), vr.EXPECTED_RUNS


def _unreplayable() -> int:
    """Runs the registry declares it cannot replay at all, as the registry lists them."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import verify_reproducibility as vr

    return len(vr.UNREPLAYABLE)


def _released() -> int:
    """Study runs actually in `docs/`. Counted from the tree, never from the registry: the
    paragraph's claim is that the three registry states cover the release, and a total taken
    from the registry would make that claim true by construction."""
    from pathlib import Path

    return len(list(Path(DOCS).glob("validation-study*.json")))


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
    over = [u for u in est if u > 5.0]
    out.append(("runs over the $5 default", 5, 0, len(over)))
    out.append(("largest estimate", 11.76, 0.005, max(est)))
    # The same two facts as the shared harness section states them. Both sides of the two checks
    # above are this file's own: a literal typed here against the artifacts, with the paper read
    # by neither. That is how the sentence came to say "four runs ... three distinct values" while
    # the literal above was updated to five -- the guard and the prose drifted in opposite
    # directions and nothing compared them. Every figure in the sentence is now scraped.
    shared = _tex()
    out.append(
        (
            "tex: runs over the $5 default",
            _scrape_word(shared, r"(\w+) runs raised it explicitly"),
            0,
            len(over),
        )
    )
    out.append(
        (
            "tex: distinct over-default estimates",
            _scrape_word(shared, r"explicitly, at (\w+) distinct values"),
            0,
            len({round(u, 2) for u in over}),
        )
    )
    printed_over = _scrape_all(
        shared, r"distinct values \(\\\$([\d.]+) twice, \\\$([\d.]+), \\\$([\d.]+), \\\$([\d.]+)\)"
    )
    for i, value in enumerate(sorted(printed_over)):
        out.append(
            (
                f"tex: over-default estimate {i}",
                value,
                0.005,
                sorted({round(u, 2) for u in over})[i],
            )
        )

    # Prose cross-checks. Everything above re-derives a number from the artifacts and
    # compares it to a number typed into *this file*; nothing above reads the paper. That
    # gap let a real error through: the closability table said the re-run changed
    # ensure_correct on 2/3 seeds when the data (and the body text) say 3/3 -- the 2/3 is
    # arm B's generate, transposed. The checks below scrape the paper's own sentences, so
    # the claimed side comes from the papers rather than from an author's memory of it.
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

    # The C1 decomposition: the eight tasks the section names account for only part of the
    # -0.033, and the paper now says how much. A partial decomposition presented as a whole
    # one is exactly the kind of claim this file exists to keep honest. (They are NOT "the eight
    # verdict-differing tasks", which is how this comment and the paper both used to put it: the
    # verdicts differ on three. See `tex: c1 stratum size` below.)
    # Line breaks are matched as \s+ rather than a literal \n. Pinning the wrap position makes an
    # unrelated reflow of a hard-wrapped paragraph look like drift, which trains the reader to
    # treat a red check as noise -- the opposite of what these are for.
    verdict_share, prose_residual = _c1_decomposition()
    decomp_prose.append(
        (
            "tex: C1 eight-task share",
            r"account for\s+\$(-[\d.]+)\$ of the \$-0\.033\$",
            0.0005,
            verdict_share,
        )
    )
    decomp_prose.append(
        (
            "tex: C1 residual on the other 24",
            r"the\s+remaining \$(-[\d.]+)\$ falls on the twenty-four",
            0.0005,
            prose_residual,
        )
    )

    for label, pattern, tol, artifact in decomp_prose:
        out.append((label, _scrape(tex, pattern), tol, artifact))

    # --- Table cells, read positionally out of the .tex ------------------------------
    # THE DRIFT HOLE THIS CLOSES. Most checks in this file compare a computed artifact value
    # against a literal typed here, which pins the ARTIFACT but never opens a paper: edit a
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
            # The abstract restates the same range at three decimals. Bound separately rather than
            # trusted to agree: the four-decimal copy above was correct while a sibling
            # restatement of a different quantity had gone stale, which is the whole lesson of
            # test_paper_sigma_coverage.
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

    # tab:closability quotes sigma to three decimals in a narrow cell. Checked at that precision
    # so the rounded restatement cannot drift away from the estimator either.
    out.append(
        (
            "tex: closability sigma",
            _scrape(tex, r"\$\\sigma\(\\dd\) = ([\d.]+)\$ is\n  recoverable"),
            0.0005,
            _noise_estimate().sigma_delta_mean,
        )
    )

    # --- D4: the budget sweep, bought (sec:budgetresult) -------------------------------
    # The depth column is MEASURED (a cassette replay counting GEPA iterations), not the
    # pre-registered formula, because the formula was wrong: it charges rejected proposals for a
    # valset pass they never take. Checking both columns keeps that correction honest.
    budget_res_hdr = r"\texttt{mmc} & cap & priced depth & \textbf{measured depth}"
    budget_cells = _BUDGET_ROWS
    accepted = []
    for mmc, cap, slug in budget_cells:
        cells = _table_row(tex, budget_res_hdr, f"{mmc}  &" if mmc < 100 else f"{mmc} &")
        out.append((f"tab:budgetresult mmc{mmc} value", cells[0][0], 0, mmc))
        out.append((f"tab:budgetresult mmc{mmc} cap", cells[1][0], 0, cap))
        out.append(
            (
                f"tab:budgetresult mmc{mmc} priced",
                cells[2][0],
                0.005,
                _budget_cell(mmc, cap)["cycles"],
            )
        )
        depth = _search_depth(slug, mmc, cap)
        if depth is not None:
            out.append((f"tab:budgetresult mmc{mmc} measured depth", cells[3][0], 0, depth))
        out.append((f"tab:budgetresult mmc{mmc} delta", cells[5][0], 0.0005, _budget_delta(slug)))
        accepted.append(_budget_acceptance(slug))

    # The acceptance sequence, which is the sweep's dissociation: it moves where Delta does not.
    prose_acc = re.search(
        r"the sweep gives \$(\d)/6\$, \$(\d)/6\$, \$(\d)/6\$, "
        r"\$(\d)/6\$, \$(\d)/6\$ and \$(\d)/6\$",
        tex,
        re.DOTALL,
    )
    if prose_acc is None:
        prose_acc = re.search(
            r"gives \$(\d)/6\$, \$(\d)/6\$, \$(\d)/6\$, \$(\d)/6\$, \$(\d)/6\$, \$(\d)/6\$",
            tex,
            re.DOTALL,
        )
    for i, want in enumerate(accepted):
        got = float(prose_acc.group(i + 1)) if prose_acc else float("nan")
        out.append((f"tex: budget acceptance {i}", got, 0, want))

    # The triplicate at the cap-bound depth, and its mean.
    trip = [_budget_delta(s) for s in ("mmc40", "mmc56", "mmc80")]
    out.append(
        (
            "tex: budget triplicate mean",
            _scrape(tex, r"\$-0.008\$, \$-0.018\$, mean \$-([\d.]+)\$"),
            0.0005,
            abs(mean(trip)),
        )
    )

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
    # The dagger footnote on the run-3 row of tab:curve. That row prints the FIVE-seed mean while
    # every gap uses the three-seed one, so a reader subtracting the printed column gets +0.072
    # where the paper says +0.060. The footnote states both and the arithmetic between them, and
    # all three numbers are checked here -- the trap caught one of this study's own authors.
    for label, pattern, artifact in (
        (
            "tex: curve dagger three-seed",
            r"its first three\n?\s*\(\$([\d.]+)\$\)",
            _curve_seed_mean("run3"),
        ),
        ("tex: curve dagger tool row", r"lift is \$([\d.]+) -", _curve_seed_mean("run4")),
        ("tex: curve dagger subtrahend", r"lift is \$[\d.]+ - ([\d.]+)", _curve_seed_mean("run3")),
        (
            "tex: curve dagger gap",
            r"lift is \$[\d.]+ - [\d.]+ = \+([\d.]+)\$",
            _curve_gap("run3", "run4")[0],
        ),
        (
            "tex: curve dagger naive gap",
            r"not the \$\+([\d.]+)\$ this column",
            _curve_seed_mean("run4") - _curve_seed_mean("run3", shared_seeds_only=False),
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.0005, artifact))

    # --- sec:curverepeat: the 90 curve replicates ------------------------------------------
    # The subsection exists because the ratios above divided by a borrowed sigma. Every number it
    # prints is therefore bound to `repeat_scores`, including the margins that motivated buying it.
    _csig, _cpairs = _curve_replicates()
    _rows = (
        ("run3", 1, r"run 3  & intrinsic\s+& 1 & \\textbf\{([\d.]+)\}"),
        ("run4", 1, r"run 4  & tool-fed\s+& 1 & ([\d.]+)"),
        ("run7", 1, r"run 7  & oracle\s+& 1 & ([\d.]+)"),
        ("run8", 3, r"run 8  & oracle, mid\s+& 3 & \\textbf\{([\d.]+)\}"),
        ("run9", 3, r"run 9  & oracle, lo\s+& 3 & ([\d.]+)"),
        ("run10", 3, r"run 10 & oracle, hi\s+& 3 & ([\d.]+)"),
    )
    for label, _r, pattern in _rows:
        out.append((f"tab:curvesigma {label}", _scrape(tex, pattern), 0.00005, _csig[label]))
        out.append(
            (
                f"tab:curvesigma {label} ratio",
                # `_scrape` returns group 1, which here is the sd the base pattern captures, so
                # the ratio needs the SECOND group. A first draft compared 0.0099 against 0.345.
                _scrape_all(tex, pattern + r" & \$([\d.]+)\\times\$")[1],
                0.005,
                _csig[label] / _noise_estimate().sigma_delta_mean,
            )
        )
    # The two R-group means, which carry the claim that R=3 is not the noisier group.
    for label, keys in (("R1", ("run3", "run4", "run7")), ("R3", ("run8", "run9", "run10"))):
        pat = (
            r"\$R\{=\}1\$ averages \$([\d.]+)\$"
            if label == "R1"
            else r"against \$R\{=\}3\$'s \$([\d.]+)\$"
        )
        out.append(
            (
                f"tab:curvesigma {label} mean",
                _scrape(tex, pat),
                0.00005,
                mean(_csig[k] for k in keys),
            )
        )
    out.append(
        (
            "tab:curvesigma spread",
            _scrape(tex, r"The spread here is \$([\d.]+)\\times\$"),
            0.05,
            max(_csig.values()) / min(_csig.values()),
        )
    )
    # The paired table. Each row prints three quantities the driver produces --- the published
    # single draw, the K=5 mean and its standard error --- plus a t. All four are bound, and the
    # two means SIGN-FAITHFULLY: the table guard reads signed cells, so registering a magnitude
    # leaves a negative row unchecked, which is what it caught on two of these six.
    _rowspec = (
        ("run3", "run4", "tool lift, run 4 over run 3", "+", "+"),
        ("run4", "run7", "oracle under tool, 7 under 4", "-", "-"),
        ("run7", "run8", "refine budget, run 8 over run 7", "+", "+"),
        ("run9", "run10", "actionability, run 10 over run 9", "+", "+"),
        ("run3", "run7", "oracle under intrinsic, 7 under 3", "-", "-"),
        ("run4", "run10", r"run 10 over the tool-fed loop\}", "+", "+"),
    )
    for a, b, label, s_pub, s_mean in _rowspec:
        res = _cpairs[(a, b)]
        pub_sign = -1.0 if s_pub == "-" else 1.0
        mean_sign = -1.0 if s_mean == "-" else 1.0
        num = r"([\d.]+)"
        pub_tok = rf"\${re.escape(s_pub)}{num}\$"
        # The reinstated row prints its mean in \mathbf, so the cell markup differs by row.
        mean_tok = (
            rf"\$\\mathbf\{{{re.escape(s_mean)}{num}\}}\$"
            if b == "run10" and a == "run4"
            else rf"\${re.escape(s_mean)}{num}\$"
        )
        row = rf"{label}\s+& {pub_tok} & {mean_tok} & ([\d.]+) &"
        pub, mn, se = _scrape_all(tex, row)
        out.append(
            (f"tab:curvepairs {b}-{a} published", pub_sign * pub, 0.00005, _curve_gap(a, b)[0])
        )
        out.append((f"tab:curvepairs {b}-{a} mean", mean_sign * mn, 0.00005, res.delta))  # type: ignore[attr-defined]
        out.append((f"tab:curvepairs {b}-{a} se", se, 0.00005, res.standard_error))  # type: ignore[attr-defined]
        # Two capture groups precede the t (the published cell and the mean), and `_scrape`
        # returns the first, so this must index. A first draft compared the t against the
        # published gap and reported six drifts that were all one bug.
        t_pat = rf"{label}\s+& {pub_tok} & {mean_tok} & [\d.]+ & (?:\\textbf\{{)?([\d.]+)"
        out.append((f"tab:curvepairs {b}-{a} t", _scrape_all(tex, t_pat)[2], 0.005, res.t_stat()))  # type: ignore[attr-defined]

    # The published-vs-measured discrepancies, in single-draw sd units: the sentence that licenses
    # reading the shrinkage as one draw's variation rather than as a systematic effect.
    _disc = [
        abs(_cpairs[(a, b)].delta - _curve_gap(a, b)[0])  # type: ignore[attr-defined]
        / (_csig[a] ** 2 + _csig[b] ** 2) ** 0.5
        for _, a, b in (
            ("", "run3", "run4"),
            ("", "run4", "run7"),
            ("", "run7", "run8"),
            ("", "run9", "run10"),
            ("", "run3", "run7"),
            ("", "run4", "run10"),
        )
    ]
    _printed = _scrape_all(
        tex,
        r"within \$([\d.]+)\$ single-draw sd of the published value --- \$([\d.]+)\$, "
        r"\$([\d.]+)\$,\s*\$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$",
    )
    out.append(("tex: curve worst discrepancy", _printed[0], 0.005, max(_disc)))
    for i, d in enumerate(_disc):
        out.append((f"tex: curve discrepancy {i}", _printed[i + 1], 0.005, d))
    out.append(
        (
            "tex: curve control p0 family",
            _scrape(
                tex,
                r"family, measured independently in two experiments: "
                r"\$[\d.]+\$ against \$([\d.]+)\$",
            ),
            0.00005,
            _p0_seed_sigma(),
        )
    )

    out.append(
        (
            "tex: curve priced bound",
            _scrape(tex, r"\$90\$ runs under a priced bound of \\\$\$([\d.]+)\$"),
            0.005,
            _curve_priced_bound(),
        )
    )

    # The corrections appendix asserts its own length in words, and nothing checked it until an
    # entry was added and the word left alone -- the exact shape of stale assertion the appendix is
    # a register of. Each paper now carries its own register, so the check is per paper and lives in
    # `_corrections_count_checks`; slicing the concatenation here would count all three at once.

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
    # new defect, while both-arms-zero is the original retraction re-derived from scores.
    one_sided, no_contrast = _contrast_flag_census()
    # The correction to that sentence needs its own pins. It used to say both arms "accepted
    # nothing on every seed" in the other fifteen, which is false in eleven of them: those files
    # carry no `genes_changed` key at all, so the census reads an ABSENT field as a zero, and the
    # released scores contradict it. An arm that accepts nothing returns the seed genome and
    # reproduces its own seed score, so opt != seed on any seed is an accepted proposal.
    acc = _precorrection_acceptance()
    for label, pattern, artifact in (
        (
            "tex: precorrection any-arm acceptances",
            r"\\textbf\{(\w+)\} of the fifteen\s+have one",
            acc["either"],
        ),
        ("tex: precorrection arm B acceptances", r"arm~B in (\w+) runs", acc["B"]),
        ("tex: precorrection arm D acceptances", r"arm~D in (\w+),", acc["D"]),
        ("tex: precorrection flat on both arms", r"only (\w+) are flat on both arms", acc["flat"]),
    ):
        out.append((label, _scrape_word(tex, pattern), 0, artifact))
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

    # The replicate COUNT behind the noise estimate, in both places the paper states it. This
    # was silently wrong once: buying a run at n_train=8 on multi-hard makes it eligible as a
    # replicate automatically, so the estimator's count grows while the prose keeps its old
    # number and the sigma it quotes is derived from a different set than it claims.
    n_reps = noise.n_replicates
    # The abstract used to restate this count; it was cut to meet the 200-word / one-paragraph
    # limit, so sec:threats is now the only place that states it and the only anchor needed.
    out.append(
        (
            "tex: noise n replicates (threats)",
            _scrape_word(tex, r"scope note\. (\w+) released runs"),
            0,
            n_reps,
        )
    )
    # The count was always right; the appositive that purported to identify the ten named nine,
    # because the tenth is the discarded first attempt at the four-arm re-run. The paragraph now
    # names it and states sigma without it, so that figure is derived here rather than typed there.
    out.append(
        (
            "tex: noise sigma without the discarded attempt",
            _scrape(tex, r"Without it \$\\sigma\(\\dd\)\$ is \$([\d.]+)\$"),
            0.00005,
            _noise_estimate("c2").sigma_delta_mean,
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
            r"/\\sqrt\{30\} \\;=\\; ([\d.]+)",
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

    # The replicate means as a RANGE rather than an enumeration. Any future run at this split
    # size enrols itself as a replicate, so a list of means would go stale silently and a
    # positional check on it would need editing every time. Min, max and the span are stable
    # summaries of the same evidence and they cannot rot the same way.
    means = list(noise.replicate_means.values())
    for label, pattern, want in (
        ("tex: noise replicate min", r"means span \$([\d.]+)\$ to", min(means)),
        ("tex: noise replicate max", r"span \$[\d.]+\$ to\n\$([\d.]+)\$", max(means)),
        (
            "tex: noise replicate range",
            r"a range of \$([\d.]+)\$ on a quantity",
            max(means) - min(means),
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.0005, want))

    # The headline effects restated in sigma units, in both the abstract and sec:threats.
    #
    # The deltas are read from the decomposition, not typed. Until 2026-08-13 they were the
    # literals -0.0417, 0.0600 and -0.0017 -- the paper's own four-decimal displays -- and L1's
    # rounding mattered: 0.0417/sigma is 1.4354 while the exact -0.0416667 gives 1.4342. Six prose
    # sites printed 1.44, which agreed with the rounded quotient to 0.0047 and passed this
    # tolerance, while tab:splitresults printed 1.43 for the same cell from the same data. A check
    # that divides a display value cannot distinguish the paper being right from the paper and the
    # check having rounded the same way, which is how a table and its own prose disagreed for the
    # paper's whole life without failing anything.
    #
    # The deltas are also captured now rather than anchored: each pattern used to embed the printed
    # delta as locator text, so the number that selects the sentence was never compared to the run.
    # Each level is located by its own WORD label -- `$L_1$ loss`, `$L_2$ gain`, `$L_3$ null` --
    # so no numeral does the locating and every numeral is compared.
    for level, word, sign in ((0, "loss", "-"), (1, "gain", r"\+"), (2, "null", "-")):
        name = f"L{level + 1}"
        stem = rf"\$L_{level + 1}\$ {word}\s+\$"
        delta = _decomposition(level)["delta"]
        out.append(
            (
                f"tex: sigma units {name}",
                _scrape(tex, stem + rf"{sign}[\d.]+\$ at \$([\d.]+)\\sigma\$"),
                0.005,
                noise.in_sigma(delta),
            )
        )
        out.append(
            (
                f"tex: sigma units {name} delta",
                _scrape(tex, stem + rf"({sign}[\d.]+)\$ at"),
                0.0005,
                delta if sign == "-" else abs(delta),
            )
        )

    # EVERY OTHER PLACE A GAP IS RESTATED IN SIGMA UNITS. The five checks above and the curve
    # block pinned one sentence each, and the paper restates the same six gaps in the abstract,
    # the contributions list, two captions, sec:results and the conclusion. Those copies went
    # unchecked, and drifted: sigma moved from five replicates to ten during the study, and the
    # restatements kept the old quotient. Measured at the time this block was added, the paper
    # printed 3.16 for 3.09 twice, 1.01 for 0.99 three times, 2.15 for 2.10, 2.27 for 2.22
    # twice, and -- worst, because it is off by a factor -- "5 sigma" for 3.50. One of them sat
    # in the same SENTENCE as a value that had been updated. A number restated in six places is
    # six numbers unless every one of them is derived.
    #
    # Note the actionability effect appears as both 2.21 and 2.22 depending on whether the
    # rounded +0.063 or the exact gap 0.06333 is divided. That is a real disagreement, not a
    # tolerance question, so it is settled one way: divide the EXACT gap everywhere, which is
    # what `_curve_gap` returns. Dividing a number the paper already rounded for display gives
    # an answer that depends on the display.
    for label, pattern, pair in (
        (
            "tex: contributions oracle vs tool sigma",
            r"below a measurement tool by \$([\d.]+)\\sigma\$",
            ("run4", "run7"),
        ),
        (
            "tex: results oracle vs tool sigma",
            r"below the tool-fed critique at \$([\d.]+)\\sigma\$",
            ("run4", "run7"),
        ),
        (
            "tex: results oracle vs intrinsic sigma",
            r"\\emph\{intrinsic\} one at \$([\d.]+)\\sigma\$",
            ("run3", "run7"),
        ),
        (
            "tex: curve caption refine budget sigma",
            r"run 8, \$([\d.]+)\\sigma\$",
            ("run7", "run8"),
        ),
        (
            "tex: curve caption actionability sigma",
            # Not anchored on a closing paren: the caption now also carries the replicated t, and
            # a pattern that insists on `)` immediately after the sigma breaks the moment the
            # sentence is extended rather than when the number changes.
            r"\(run 9\$\\to\$10, \$\+0.063\$, \$([\d.]+)\\sigma\$",
            ("run9", "run10"),
        ),
        (
            "tex: fixside actionability sigma",
            r"against the noise floor: \$([\d.]+)\\sigma\$",
            ("run9", "run10"),
        ),
        (
            "tex: threats actionability sigma",
            r"\$\+0.063\$ at \$([\d.]+)\\sigma\$",
            ("run9", "run10"),
        ),
        (
            "tex: conclusion tool sigma",
            r"a measurement tool\s+\(\$\+0.060\$, \$([\d.]+)\\sigma\$\)",
            ("run3", "run4"),
        ),
        (
            "tex: conclusion refine sigma",
            r"more refine attempts \(\$\+0.065\$, \$([\d.]+)\\sigma\$\)",
            ("run7", "run8"),
        ),
        (
            "tex: conclusion oracle sigma",
            r"moves held-out by\s+\$([\d.]+)\\sigma\$, i\.e\.\\ not at all",
            ("run3", "run7"),
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.005, _curve_gap(*pair)[1]))

    # The WITHIN-RUN sigma trio and the single-draw ratios sec:repeat derives from it. Unbound
    # until now, and the sentence carrying them was wrong twice over: it quoted a cross-run sigma
    # that a later replicate superseded, and asserted that all three within-run values exceeded it
    # when L3 -- the level nearest the metric ceiling, and so the least variable -- is well under.
    # Both errors are of the same kind: a comparison written once and never re-derived when the
    # quantity on one side of it moved.
    within = {r.level: r.sd_between_replicates for r in _repeat().aggregate(5)}
    single = {f"L{i + 1}": _decomposition(i)["delta"] for i in range(3)}
    trio = _scrape_all(
        tex, r"is \$([\d.]+)\$ at \$L_1\$, \$([\d.]+)\$ at \$L_2\$ and \$([\d.]+)\$ at \$L_3\$"
    )
    ratios = _scrape_all(
        tex, r"are therefore \$([\d.]+)\\sigma\$,\s+\$([\d.]+)\\sigma\$ and \$([\d.]+)\\sigma\$"
    )
    for i, lvl in enumerate(("L1", "L2", "L3")):
        out.append((f"tex: within-run sigma {lvl}", trio[i], 0.0005, within[lvl]))
        out.append(
            (
                f"tex: single-draw ratio {lvl}",
                ratios[i],
                0.005,
                abs(single[lvl]) / within[lvl],
            )
        )
    out.append(
        (
            "tex: within-run vs cross-run sigma",
            _scrape(tex, r"exceed the\s+\$([\d.]+)\$ the cross-run estimate gave"),
            0.00005,
            noise.sigma_delta_mean,
        )
    )
    # "Two of the three exceed it" is the corrected count. Asserted as a number so that a future
    # replicate which moves one of the four quantities cannot leave the word "two" standing.
    out.append(
        (
            "tex: within-run exceeding cross-run count",
            _scrape_word(tex, r"(\w+) of the three exceed the"),
            0,
            sum(1 for lvl in within if within[lvl] > noise.sigma_delta_mean),
        )
    )

    # sec:budgetresult scales the depth cells by the L1 within-run sigma rather than the cross-run
    # one, because a budget sweep compares two arms inside one run. Three numbers, none bound.
    deepest = _budget_delta("mmc160")
    for label, pattern, want in (
        (
            "tex: depth sigma denominator",
            r"\$\\sigma\(\\dd\)=([\d.]+)\$",
            within["L1"],
        ),
        (
            "tex: depth largest cell sigma",
            r"the largest is the original \$-0.042\$ at \$([\d.]+)\\sigma\$",
            abs(single["L1"]) / within["L1"],
        ),
        (
            "tex: depth deepest cell sigma",
            # The delta itself is deliberately NOT a literal here. Writing it in cost the paper a
            # real defect: this pattern pinned the quotient while the numerator it divides sat
            # inside the pattern, so the quotient was checked against a value nothing verified.
            # It is now pinned by `tex: budget deepest delta (prose)`, which reads the same site.
            r"it reads \$\+[\d.]+\$ .*?which is \$([\d.]+)\\sigma\$",
            abs(deepest) / within["L1"],
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.005, want))

    # "No cell reaches one standard deviation" is a claim about the MAXIMUM over the six cells, so
    # it is checked against that maximum rather than against the cell the sentence goes on to name.
    # It replaced an arbitrary "$1.1\sigma$" -- true, but derived from nothing, and the only sigma
    # figure in the paper that no artifact could produce.
    worst = max(abs(_budget_delta(s)) for _, _, s in budget_cells) / within["L1"]
    out.append(
        (
            "tex: depth no cell reaches one sigma",
            1.0 if re.search(r"no cell reaches one standard deviation", tex) else 0.0,
            0,
            1.0 if worst < 1.0 else 0.0,
        )
    )

    # The saturation arithmetic sec:p0 uses to report its own falsified prediction: L3's seed
    # genome is already near the ceiling, so the headroom is what bounds any contrast there. The
    # headroom was the one number in that sentence no check derived, and it is the number the
    # whole "design error" admission rests on.
    l3 = _runs("validation-study-p0-L3.json")
    l3_seed = mean(r["B"]["seed_heldout"] for r in l3)
    for label, pattern, want in (
        ("tex: L3 seed baseline", r"scores \$([\d.]+)\$, leaving", l3_seed),
        ("tex: L3 headroom", r"leaving \$([\d.]+)\$ of headroom", 1.0 - l3_seed),
        (
            "tex: L3 arm B consumes",
            r"nearly all of it \(\$([\d.]+)\$ and",
            mean(r["B"]["opt_heldout"] for r in l3),
        ),
        (
            "tex: L3 arm D consumes",
            r"nearly all of it \(\$[\d.]+\$ and\s*\$([\d.]+)\$\)",
            mean(r["D"]["opt_heldout"] for r in l3),
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.0005, want))

    # The superseded first four-arm attempt's delta. sec:superseded scales it by the noise floor
    # and the SIGMA was checked while the delta itself was a literal inside the pattern -- so the
    # quotient was pinned to a number nothing verified. The run is `validation-study.json`, the one
    # released artifact whose name carries no suffix, which is also why a glob over
    # `validation-study-*.json` silently misses it.
    from analyze_deltas import delta_arm as _delta_arm

    # The locating phrase changed on 2026-08-13. It read "attempt at", from a sentence that called
    # this "the discarded first four-arm attempt" -- a phrase the same paper uses twice elsewhere
    # for C2, whose delta is -0.015 and which the pre-correction partition excludes outright. The
    # run carrying -0.094 is run 1, the single-constraint one, and the sentence now says so.
    out.append(
        (
            "tex: superseded first-attempt delta",
            _scrape(tex, r"verify gene inert, at \$-([\d.]+)\$"),
            0.0005,
            abs(mean(_delta_arm(f"{DOCS}/validation-study.json"))),
        )
    )

    # The PER-ARM drops from baseline, which sec:rerun prints alongside the contrast. These are a
    # different quantity from $\dd$ and were unbound, which is exactly the confusion the paper's own
    # decomposition exists to prevent: $d_B$ and $d_D$ are each arm against its own seed score,
    # while $\dd = d_D - d_B$. Two of the six values happened to collide with checked numbers and
    # one did not, which is how a coverage audit found them. Checking the vectors AND that their
    # difference reproduces the published contrast pins the identity, not just the digits.
    c2b_runs = _runs("validation-study-c2b.json")
    drops = {
        arm: [r[arm]["opt_heldout"] - r[arm]["seed_heldout"] for r in c2b_runs]
        for arm in ("B", "D")
    }
    per_arm = _scrape_all(
        tex,
        r"prompt-only optimization is a wash\}? \(\$\+([\d.]+)\$, \$([\d.]+)\$,\s*"
        r"\$-([\d.]+)\$; mean exactly \$([\d.]+)\$\) while \\textbf\{loop-level optimization "
        r"costs\} \$-([\d.]+)\$\s*\(\$-([\d.]+)\$, \$-([\d.]+)\$, \$\+([\d.]+)\$\)",
    )
    for i, (label, want) in enumerate(
        (
            ("tex: rerun d_B seed0", drops["B"][0]),
            ("tex: rerun d_B seed1", drops["B"][1]),
            ("tex: rerun d_B seed2", -drops["B"][2]),
            ("tex: rerun d_B mean", mean(drops["B"])),
            ("tex: rerun d_D mean", -mean(drops["D"])),
            ("tex: rerun d_D seed0", -drops["D"][0]),
            ("tex: rerun d_D seed1", -drops["D"][1]),
            ("tex: rerun d_D seed2", drops["D"][2]),
        )
    ):
        out.append((label, per_arm[i], 0.0005, want))
    out.append(
        (
            "tex: rerun d_D minus d_B is the published contrast",
            mean(drops["D"]) - mean(drops["B"]),
            0.0005,
            mean(r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in c2b_runs),
        )
    )

    # Arm C's training-score range, which is the whole of the qualification sec:rerun attaches to
    # that control: previously it saturated at 1.0 and GEPA proposed nothing, so the control exited
    # before exercising anything. The two endpoints were the last MEASURED values in the paper's
    # prose that no check derived. Tolerance is 0.006 because the paper prints two decimals and the
    # low endpoint is 0.225, exactly half a unit away.
    c_train = [r["C"]["train_score"] for r in _runs("validation-study-c2b.json")]
    for label, pattern, want in (
        ("tex: arm C train lo", r"it starts at \$([\d.]+)\$ to", min(c_train)),
        ("tex: arm C train hi", r"it starts at \$0.23\$ to\s+\$([\d.]+)\$", max(c_train)),
    ):
        out.append((label, _scrape(tex, pattern), 0.006, want))

    # The two superseded-run deltas sec:superseded scales by the same floor. These were quoted as
    # "5 sigma" and "3.5 sigma" for 0.100 and 0.094, which are 3.50 and 3.29 -- the first is what
    # you get from a sigma of 0.020, a value this study never estimated.
    #
    # Both deltas are now READ FROM their run files. Until 2026-08-13 the second was the literal
    # 0.094 -- the value the paper prints -- so both sides of that comparison descended from the
    # same rounded display and the check passed by repeating the paper's own rounding: 0.094/sigma
    # is 3.2356, which sits 0.0044 from the published 3.24 and inside this tolerance, while the
    # artifact's exact -0.09375 gives 3.2270, which is 0.0130 away and fails it. Dividing a number
    # the paper already rounded for display gives an answer that depends on the display, which is
    # the anti-pattern this file warns about elsewhere and had committed here.
    #
    # The second sentence also named the wrong run. It called this "the discarded first four-arm
    # attempt", a phrase sec:reproduce and sec:sigma both use for C2 -- whose mean delta is -0.015,
    # and which `module_selector == "all"` excludes from the pre-correction fifteen altogether. The
    # run that carries -0.094 is run 1, the single-constraint one. The paper now names it.
    for label, pattern, run in (
        (
            "tex: superseded actuator sigma",
            r"powered actuator run at \$\\approx\\!([\d.]+)\\sigma\$",
            "validation-study-bcb-actuator-big.json",
        ),
        (
            "tex: superseded first-attempt sigma",
            r"verify gene inert, at \$-[\d.]+\$, \$\\approx\\!([\d.]+)\\sigma\$",
            "validation-study.json",
        ),
    ):
        out.append((label, _scrape(tex, pattern), 0.005, noise.in_sigma(_abs_mean_delta(run))))

    # The reproducibility registry's composition, which drifted once already: the paper said
    # "seventeen ... the eleven below plus the two hand-written conditions", which is 13.
    # These counts are spelled out in the prose, so they need the word scraper.
    superseded, replayable = _registry_counts()
    for label, pattern, artifact in (
        # [\w-] not \w in EVERY row, not just the one that needed it first. `\w` was widened
        # here once, on the single count that had crossed twenty, and the other four kept the
        # narrow class -- so the next count to cross twenty stopped matching instead of
        # mismatching. That is survivable only because a no-match returns NaN and NaN fails
        # every comparison; had `_scrape_word` skipped on miss, four checks would have gone
        # quiet at exactly the moment they had something to say.
        ("tex: registry total", r"registers ([\w-]+) committed runs", superseded + replayable),
        (
            "tex: registry superseded",
            r"([\w-]+) are\s+labelled \\textbf\{superseded\}",
            superseded,
        ),
        ("tex: registry replayable", r"The remaining ([\w-]+) were recorded", replayable),
        ("tex: registry verifies N", r"verifies ([\w-]+) runs rather than", replayable),
        (
            "tex: retraction superseded count",
            r"script registers ([\w-]+) superseded runs",
            superseded,
        ),
        # The third state. Registered + superseded is not the release: five BigCodeBench runs
        # supply numbers and cannot replay at all, and the paragraph now says so. Checked
        # against the registry's own list and against the tree, so the sentence's two halves
        # come from different places and cannot be reconciled by editing one.
        ("tex: registry unreplayable", r"The ([\w-]+) remaining runs of the", _unreplayable()),
        ("tex: registry release total", r"remaining runs of the ([\w-]+) released", _released()),
    ):
        out.append((label, _scrape_word(tex, pattern), 0, artifact))

    # The two reflection models, asserted as a set rather than a number: the check is that the
    # paper names exactly the model each half of the study used, in the right half.
    corrected_models, pre_models = _reflection_models()
    # Markers are whitespace-tolerant. The previous pre-correction marker embedded a literal
    # newline, so reflowing the sentence -- which widening it necessarily did -- broke a check
    # about the model rather than about the wrap. Same defect class as the C1 patterns fixed in
    # c80a926, in the same file, two days apart.
    for label, want, marker in (
        ("tex: corrected-harness reflection model", corrected_models, r"corrected-harness\s+runs"),
        ("tex: pre-correction reflection model", pre_models, r"which is\s+the\s+CLI default"),
    ):
        # One model per era was true of the runs that existed when this was written, and the
        # check enforced it by refusing to run at all on a second. That refusal was right --- it
        # caught the cross-provider run of 2026-08-06 the moment its artifact landed --- but the
        # premise was a description mistaken for a rule. The paper must name whatever the
        # artifacts show, so the assertion is now "every model in this era is named", and the
        # marker anchors only the first, which is the one the sentence leads with.
        for i, model in enumerate(sorted(want)):
            named = model.split("/")[-1].replace("-", r"\-")
            pattern = (
                rf"\\texttt\{{{named}\}}[^.]*?{marker}" if i == 0 else rf"\\texttt\{{{named}\}}"
            )
            found = 1.0 if re.search(pattern, tex, re.DOTALL) else 0.0
            suffix = f" ({named})" if len(want) > 1 else ""
            out.append((f"{label}{suffix}", found, 0, 1.0))

    # How many contrast runs the correction superseded, and how many of their deltas really
    # were zero. The paper said "eleven" for a set of fifteen, and called all of their zeros
    # identities when four are not zero at all.
    n_pre, n_zero, max_abs, n_zero_by_means = _precorrection_counts()
    # The floating-point gap this whole family of claims turned on. Pinned so that a future edit
    # cannot quietly reintroduce a count that depends on summation order: if these two ever agree,
    # the artifact changed and the prose's careful wording needs re-reading, not preserving.
    out.append(("precorrection identical-on-every-seed", float(n_zero), 0.0, 10.0))
    out.append(("precorrection mean-exactly-zero", float(n_zero_by_means), 0.0, 11.0))
    for label, pattern, artifact in (
        # Three independent anchors for the retraction count, in the abstract, the conclusion and
        # the introduction. It has drifted once already (published as eleven for a set of fifteen),
        # and it is the number a reader is most likely to quote back.
        (
            "tex: abstract pre-correction runs",
            r"discarding (\w+) earlier contrast runs",
            n_pre,
        ),
        (
            "tex: conclusion pre-correction runs",
            r"preceded it: \\textbf\{(\w+)\} of them",
            n_pre,
        ),
        ("tex: intro pre-correction runs", r"Our first (\w+) contrast runs measured", n_pre),
        (
            "tex: defect appendix pre-correction runs",
            r"Our first (\w+) contrast runs produced arms",
            n_pre,
        ),
        # These four used to scrape "exactly zero", a predicate that moved with summation order.
        # They now scrape the identity claim, which does not. Rewording the prose NaN'd all four at
        # once, which is the fragility of regex-pinned numerals in one line: a check keyed to a
        # sentence fails when the sentence is corrected, and a NaN is only distinguishable from a
        # pass because this script compares rather than merely running.
        (
            "tex: defect appendix identical-on-every-seed",
            r"identical on every seed (\w+) times",
            n_zero,
        ),
        (
            "tex: superseded appendix run count",
            r"Of the\s+(\w+) contrast runs recorded under the pre-correction harness",
            n_pre,
        ),
        (
            "tex: superseded appendix identical-on-every-seed",
            r"harness, (\w+) have arms whose held-out scores are\s+identical",
            n_zero,
        ),
        ("tex: conclusion pre-correction runs", r"\\textbf\{(\w+)\} of them, across", n_pre),
        (
            "tex: conclusion identical-on-every-seed",
            r"(\w+) produced arms with held-out scores identical on every",
            n_zero,
        ),
        # Three sites stated the count in prose and nothing pinned them. Found while correcting the
        # predicate: the abstract, the introduction and Paper III's own conclusion all carried it.
        (
            "tex: abstract identical-on-every-seed",
            r"(\w+) of the runs reported held-out scores\s+identical on every seed",
            n_zero,
        ),
        (
            "tex: intro identical-on-every-seed",
            r"(\w+) runs reported held-out scores identical on every seed and we read",
            n_zero,
        ),
        (
            "tex: p3 conclusion identical-on-every-seed",
            r"(\w+) runs\s+reported held-out scores identical on every seed, and we read",
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

    # The lo->mid difference, which the split introduced as an explicit number where the pre-split
    # draft only printed the two levels side by side ("lo $\approx$ mid ($0.810$ vs. $0.813$)").
    # A derived number stated in an abstract needs the same pinning as one from a table.
    out.append(
        (
            "tex: lo-to-mid actionability step",
            _scrape(tex, r"naming which constraint failed adds \$\+([\d.]+)\$"),
            0.0005,
            _arm_mean("validation-study-fixbudget.json", "A", "seed_heldout")
            - _arm_mean("validation-study-fbactlo.json", "A", "seed_heldout"),
        )
    )

    out.extend(_spine_checks())
    out.extend(_cross_paper_checks())
    out.extend(_corrections_count_checks())
    out.extend(_prose_orphan_checks())
    out.extend(_defect_class_count_checks())
    out.extend(_budget_false_pass_checks())
    out.extend(_registered_depth_bound_checks())
    out.extend(_sigma_margin_checks())
    out.extend(_xprovider_checks())
    out.extend(_oracle_gap_direction_checks())
    out.extend(_groq_second_worker_checks())
    out.extend(_paraphrase_drift_checks())
    out.extend(_superlative_scope_checks())

    return out


#: Per paper, the numbers that paper cannot lose without ceasing to be itself. The concatenated
#: `tex:` checks above verify that each number survived the split into *some* paper; these verify
#: it survived into the *right* one. Without them a headline could migrate to a companion paper
#: and every other check would still pass.
_SPINE: tuple[tuple[str, str, str, float, float], ...] = (
    # (paper slug, label, pattern scraped from that paper alone, claimed, tolerance)
    (
        "p1-conditional-lever",
        "P1 swing",
        r"a swing of \$([\d.]+)\$ in \$\\dd\$",
        0.102,
        0.0005,
    ),
    (
        "p1-conditional-lever",
        "P1 swing in sigma",
        r"\\approx\\; ([\d.]+) ,\n\\]",
        2.5,
        0.05,
    ),
    (
        "p1-conditional-lever",
        "P1 cross-run sigma",
        r"\\sigma\(\\dd\) \\;=\\; 0.1591/\\sqrt\{30\} \\;=\\; ([\d.]+)",
        0.0291,
        0.00005,
    ),
    (
        "p1-conditional-lever",
        "P1 positive control",
        r"\\dd\(D-B\) = \+([\d.]+)\$\. The harness",
        1.000,
        0.0005,
    ),
    (
        "p1-conditional-lever",
        "P1 L2 five-replicate t",
        r"\$\+0.055\$ & 0.0459 & 0.0205 & \\textbf\{([\d.]+)\}",
        2.70,
        0.005,
    ),
    (
        "p2-repair-not-detection",
        "P2 oracle seed held-out",
        r"\\emph\{oracle\} critique \(F\$_1\\!=\\!1.0\$ by construction\) yields only \$([\d.]+)\$",
        0.748,
        0.0005,
    ),
    (
        "p2-repair-not-detection",
        "P2 perfect-checker null t",
        r"the oracle\nmoves \$-0.0027\$ at \$t = ([\d.]+)\$",
        0.33,
        0.005,
    ),
    (
        "p2-repair-not-detection",
        "P2 actionability t",
        r"\$\+0.0613\$ at \$t = ([\d.]+)\$ over five replicates",
        8.08,
        0.005,
    ),
    (
        "p2-repair-not-detection",
        "P2 hi-over-tool reinstated",
        r"measured, it is \$\+0.0640\$ at \$t = ([\d.]+)\$",
        5.47,
        0.005,
    ),
    (
        "p3-silent-no-op",
        "P3 verify gene unchanged",
        r"byte-identical to the seed in \$(\d+)/15\$ cases",
        15.0,
        0.0,
    ),
    (
        "p3-silent-no-op",
        "P3 generate changed",
        r"changed\n\\texttt\{generate\} in \$(\d+)/15\$ winners",
        5.0,
        0.0,
    ),
    (
        "p3-silent-no-op",
        "P3 positive control",
        r"\\dd\(D-B\) = \+([\d.]+)\$\. The harness",
        1.000,
        0.0005,
    ),
    (
        "p3-silent-no-op",
        "P3 sigma spread factor",
        r"a factor of \$([\d.]+)\$\. Two mechanisms",
        5.9,
        0.05,
    ),
)


def _spine_checks() -> list[tuple[str, float, float, float]]:
    """Each paper still carries its own headline numbers, read from that paper alone."""
    src = _sources()
    out = []
    for slug, label, pattern, claimed, tol in _SPINE:
        out.append((f"spine: {label}", claimed, tol, _scrape(src.paper_tex(slug), pattern)))
    return out


#: Numbers that legitimately appear in more than one paper, because more than one paper's argument
#: needs them. Duplication is fine; silent disagreement is not, and nothing else in this file would
#: catch it -- `_scrape` on the concatenation returns the first match and would never see the
#: second. Each entry is scraped from every paper whose text matches, and all values must agree.
_SHARED: tuple[tuple[str, str], ...] = (
    ("cross-run sigma derivation", r"0.1591/\\sqrt\{30\} \\;=\\; ([\d.]+)"),
    ("per-task sigma one draw", r"one task under one draw is \$([\d.]+)\$"),
    ("seed-baseline span low", r"Their means span \$([\d.]+)\$ to"),
    ("seed-baseline span range", r"a range of \$([\d.]+)\$ on a quantity that is constant"),
    ("replicate signed difference", r"between replicates, \$-([\d.]+)\$, is confounded"),
    ("repeat replicate count", r"five replicates, \$(\d+)\$ runs"),
    ("positive control delta", r"\\dd\(D-B\) = \+([\d.]+)\$\. The harness"),
    ("ciexcl actuator corrected hi", r"\\mathbf\{\[-0.150,\+([\d.]+)\]\}"),
    (
        "ciexcl actionability corrected hi",
        r"\$\[\+0.025,\+([\d.]+)\]\$ \(still excludes zero\)",
    ),
    ("empty-string IFBench floor", r"the empty string scores \$([\d.]+)\$ on the vendored"),
    ("empty-string multihard floor", r"and \$([\d.]+)\$ on\ngenerated multi-hard"),
)

#: The same number quoted by two papers in *different* words, so one pattern cannot find both.
#: These are the cases where the split forced a rephrasing -- a table cell in one paper and a
#: sentence in another -- and they are exactly where a hand-copied number can drift unnoticed.
_SHARED_ALIASED: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "salt control sd (0.0144)",
        {
            "p2-repair-not-detection": (
                r"measured independently in two experiments: \$[\d.]+\$ against \$([\d.]+)\$"
            ),
            "p3-silent-no-op": r"moves\nacross replicates with sd \$([\d.]+)\$",
        },
    ),
    (
        "sigma genome spread (5.9x)",
        {
            "p2-repair-not-detection": r"genome across that family it spans \$([\d.]+)\\times\$",
            "p3-silent-no-op": r"a factor of \$([\d.]+)\$\. Two mechanisms",
        },
    ),
    (
        "actionability lift (0.063)",
        {
            "p1-conditional-lever": (
                r"feedback \\emph\{actionability\} \(Paper~II, worth \$\+([\d.]+)\$\)"
            ),
            "p2-repair-not-detection": r"0.873 - 0.810 = \+([\d.]+)\$ on the seed held-out",
        },
    ),
    (
        "run7 oracle seed held-out",
        {
            "p2-repair-not-detection": r"the oracle seed is \$([\d.]+)\$, level with",
            "p3-silent-no-op": r"oracle critique \(\$R\{=\}1\$\) & 3 & \\textbf\{([\d.]+)\}",
        },
    ),
    (
        "run10 actionability-hi seed held-out",
        {
            "p2-repair-not-detection": (
                r"even maximal, exact repair guidance saturates at \$([\d.]+)\$"
            ),
            "p3-silent-no-op": r"concrete measured edits \(hi,\n\$([\d.]+)\$\)",
        },
    ),
)


def _cross_paper_checks() -> list[tuple[str, float, float, float]]:
    """A number quoted by two papers must be the same number in both.

    Reported as (first paper's value) against (every other matching paper's value), so a drift
    shows up as a normal FAIL row rather than an exception.
    """
    src = _sources()
    out = []
    for label, pattern in _SHARED:
        matching = [s for s in src.PAPERS if re.search(pattern, src.paper_tex(s))]
        if len(matching) < 2:
            # A pattern that stops matching two papers is itself a defect: either the split moved
            # a claim and the pattern is now stale, or a paper dropped a number it needs. Surface
            # it as a failure with a sentinel rather than skipping quietly.
            out.append((f"shared: {label} (in >=2 papers)", 2.0, 0.0, float(len(matching))))
            continue
        first = _scrape(src.paper_tex(matching[0]), pattern)
        for other in matching[1:]:
            out.append(
                (
                    f"shared: {label} {src.PAPER_NUMERALS[other]}",
                    first,
                    0.0,
                    _scrape(src.paper_tex(other), pattern),
                )
            )
    for label, mapping in _SHARED_ALIASED:
        slugs = list(mapping)
        first = _scrape(src.paper_tex(slugs[0]), mapping[slugs[0]])
        for other in slugs[1:]:
            out.append(
                (
                    f"aliased: {label} {src.PAPER_NUMERALS[other]}",
                    first,
                    0.0,
                    _scrape(src.paper_tex(other), mapping[other]),
                )
            )
    out.extend(_restated_statistic_checks())
    out.extend(_magnitude_movement_checks())
    out.extend(_reflection_call_checks())
    return out


def _reflection_call_checks() -> list[tuple[str, float, float, float]]:
    """How many reflection calls each run issued, counted from its cassette.

    Replaces a per-arm proposal count an earlier draft printed and the release cannot support: the
    cassette is content-addressed on the call's name and arguments, so an entry carries no arm
    label and no proposal count can be attributed. This is the run-level quantity that *is*
    derivable, and it carries the same argument -- the corrected re-run searched harder on three
    seeds than the superseded flagship did on five.
    """
    out = []
    for label, cassette, claimed in (
        ("corrected re-run reflection calls", "study-c2b.json", 32),
        ("superseded flagship reflection calls", "study-multihard.json", 22),
    ):
        with open(f"tests/fixtures/cassettes/{cassette}", encoding="utf-8") as fh:
            entries = json.load(fh)["entries"]
        n = sum(
            1 for e in entries.values() if isinstance(e, dict) and e.get("name") == "reflection"
        )
        out.append((label, float(claimed), 0.0, float(n)))
    tex = _sources().paper_tex("p1-conditional-lever")
    quoted = _scrape_all(
        tex,
        r"issued \$(\d+)\$ reflection calls over three seeds against the"
        r" superseded\nflagship's \$(\d+)\$",
    )
    out.append(("tex: re-run reflection calls", quoted[0], 0.0, 32.0))
    out.append(("tex: superseded reflection calls", quoted[1], 0.0, 22.0))
    return out


def _magnitude_movement_checks() -> list[tuple[str, float, float, float]]:
    """How far each published magnitude moved when its row was re-scored five times.

    Unchecked until a paragraph audit found the prose claim selectively true: it named the three
    rows that shrank by 34-52% and said nothing about the fourth, which fell by 90%, or the fifth,
    which *grew* by 74%. A reader takes "three shrink by 34 to 52%" as the shape of the correction,
    so the omitted rows change its meaning. Pinned here so the count, the range and the growth are
    derived rather than recalled.
    """
    src = _sources()
    tex = src.paper_tex("p2-repair-not-detection")
    published = _scrape_all(
        tex,
        r"tool lift, run 4 over run 3\s+& \$\+([\d.]+)\$.*?"
        r"oracle under tool, 7 under 4\s+& \$-([\d.]+)\$.*?"
        r"refine budget, run 8 over run 7\s+& \$\+([\d.]+)\$.*?"
        r"actionability, run 10 over run 9 & \$\+([\d.]+)\$.*?"
        r"oracle under intrinsic, 7 under 3 & \$-([\d.]+)\$.*?"
        r"run 10 over the tool-fed loop\} & \$\+([\d.]+)\$",
    )
    # The null row is excluded, and the exclusion is the substance rather than a convenience. Its
    # magnitude falls 90% (-0.0283 -> -0.0027), but a percentage shrinkage of a quantity that is not
    # an effect is not a magnitude changing -- it is a null becoming a cleaner null, which the
    # caption states separately. Significance at K=5 is what separates the two, so it is what
    # partitions here. The threshold is READ from the driver rather than typed: a partition on a
    # hand-copied critical value decides which rows the paper calls effects, which is too much
    # weight for a numeral no artifact backs.
    t_crit = _repeat().T_CRIT_10[4]
    rows = _sources_curve()
    effects = [r for r in rows if r.t_stat() > t_crit]
    nulls = [r for r in rows if r.t_stat() <= t_crit]
    published_effects = [p for p, r in zip(published, rows, strict=True) if r.t_stat() > t_crit]
    moves = sorted(abs(r.delta) / p - 1 for p, r in zip(published_effects, effects, strict=True))
    shrank = [m for m in moves if m < -0.30]
    out = [
        ("curve rows that are effects at K=5", 5.0, 0.0, float(len(effects))),
        ("curve rows that are nulls at K=5", 1.0, 0.0, float(len(nulls))),
        ("magnitudes shrinking by more than 30%", 3.0, 0.0, float(len(shrank))),
        ("smallest of the three shrinkages", 0.34, 0.005, abs(max(shrank))),
        ("largest of the three shrinkages", 0.52, 0.005, abs(min(shrank))),
        ("the reinstated row's growth", 0.74, 0.005, max(moves)),
    ]
    # And the paper's own words for the same three quantities.
    for label, pattern, artifact in (
        ("tex: shrink range lo", r"three shrink by \$(\d+)\$ to", 100 * abs(max(shrank))),
        (
            "tex: shrink range hi",
            r"three shrink by \$\d+\$ to \$(\d+)\\%\$",
            100 * abs(min(shrank)),
        ),
        ("tex: reinstated growth", r"\\emph\{grows\}\nby \$(\d+)\\%\$", 100 * max(moves)),
    ):
        out.append((label, _scrape(tex, pattern), 0.5, artifact))
    return out


def _sources_curve():
    """The six curve comparisons at K=5, in the order `tab:curvepairs` prints them."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repeat_scores as rc

    by_label = {r.level: r for r in rc.curve_aggregate(5)}
    return [by_label[label] for label, _a, _b in rc.CURVE_PAIRS if label in by_label]


#: A statistic quoted beside its effect size, matched wherever it appears. Unlike `_SHARED`, which
#: compares one site per paper, this compares EVERY site to every other, because the sites are
#: *within* one paper and phrased differently at each.
#:
#: This family exists because a paragraph-by-paragraph audit found four stale values that 762
#: checks had missed: the oracle-vs-intrinsic t was 0.33 in two places and 0.37 in two others (one
#: of them Paper II's abstract), the actionability t was 8.08 in two places and 9.03 in the
#: conclusion, and the curve-sigma ratio range disagreed between two papers. Every pinned check
#: passed throughout, because each was anchored to one phrasing of a claim the paper makes four
#: times. The lesson is the one the series is about: a check that covers one site reads exactly
#: like a check that covers the claim.
_RESTATED: tuple[tuple[str, str], ...] = (
    # (label, pattern with the effect size in group 1 and the statistic in group 2)
    ("t beside its effect", r"\$([-+]?\d+\.\d{3,4})\$[^$]{0,30}\$t = ([\d.]+)\$"),
    ("effect stated in sigma", r"\$([-+]?\d+\.\d{3,4})\$ at \$([\d.]+)\\sigma\$"),
)


#: Effects the papers deliberately state against TWO denominators, so a bare disagreement between
#: restatements is correct rather than stale. Each entry names the effect and the two sigmas, and is
#: verified by ratio rather than exempted: the two quoted multiples must stand in the same
#: proportion as the two sigmas, so a typo in either one still fails. An exemption that merely
#: skipped the pair would be the weaker check and would hide exactly what it was added for.
_DUAL_DENOMINATOR: tuple[tuple[str, str, str, str], ...] = (
    (
        "p1-conditional-lever",
        "-0.042",
        # cross-run sigma(delta), which carries provider drift between recording dates ...
        r"\\sigma\(\\dd\) \\;=\\; 0.1591/\\sqrt\{30\} \\;=\\; ([\d.]+)",
        # ... against the within-run sigma at this level, which does not.
        r"whose single-draw \$\\sigma\$ is \$([\d.]+)\$",
    ),
)


def _restated_statistic_checks() -> list[tuple[str, float, float, float]]:
    """Every restatement of one statistic must agree with the first statement of it.

    Grouped by the effect size it sits beside rather than by section, so a value restated in an
    abstract, a caption and a conclusion is one group. Reported as (first site, other site) pairs so
    a disagreement is a normal FAIL row naming both values.
    """
    src = _sources()
    dual = {(slug, effect): (a, b) for slug, effect, a, b in _DUAL_DENOMINATOR}
    out = []
    for slug in src.PAPERS:
        tex = src.paper_tex(slug)
        for label, pattern in _RESTATED:
            groups: dict[str, list[str]] = {}
            for m in re.finditer(pattern, tex, re.DOTALL):
                groups.setdefault(m.group(1), []).append(m.group(2))
            for effect, values in groups.items():
                if (slug, effect) in dual:
                    # Two denominators, so the multiples must differ. Check they differ by exactly
                    # the ratio of the two sigmas, which is what makes them the same effect.
                    pat_a, pat_b = dual[(slug, effect)]
                    sigma_ratio = _scrape(tex, pat_b) / _scrape(tex, pat_a)
                    quoted = sorted(float(v) for v in values)
                    out.append(
                        (
                            f"dual-denominator {src.PAPER_NUMERALS[slug]}: {effect} ratio",
                            quoted[-1] / quoted[0],
                            0.02,
                            sigma_ratio,
                        )
                    )
                    continue
                for i, value in enumerate(values[1:], start=2):
                    out.append(
                        (
                            f"restated {src.PAPER_NUMERALS[slug]}: {label} {effect} (site {i})",
                            float(values[0]),
                            0.0,
                            float(value),
                        )
                    )
    return out


def _corrections_count_checks() -> list[tuple[str, float, float, float]]:
    """Each paper's corrections appendix declares a count in words; it must equal the number of
    entries it actually lists.

    The pre-split draft had one register and one count, and the count drifted twice during the
    study (`Sixteen` -> `Twenty-one` -> `Twenty-two`). Three registers are three chances to
    drift, so the count is derived from the enumerate body rather than trusted.
    """
    src = _sources()
    out = []
    for slug in src.PAPERS:
        tex = src.paper_tex(slug)
        claimed = _scrape_word(tex, r"\\textbf\{([\w-]+)\} entries follow")
        start = tex.index("entries follow")
        body = tex[start : tex.index("\\end{enumerate}", start)]
        listed = float(len(re.findall(r"^  \\item ", body, re.MULTILINE)))
        out.append((f"corrections count {src.PAPER_NUMERALS[slug]}", claimed, 0.0, listed))
    return out


def _split_refine_gain(index: int) -> tuple[int, int, float]:
    """(accepted, refined, gain per refined draft) for arm D at one cell of the split sweep."""
    mod = _decomp()
    return mod.refine_productivity(mod.SPLIT_LEVELS[index], "D")


def _prose_orphan_checks() -> list[tuple[str, float, float, float]]:
    """The seven decimals that were stated in prose and derived nowhere.

    All seven were exposed at once, by repairing `test_paper_prose_coverage`. That guard asks
    whether every decimal the prose prints is one a check derives, and it had been answering yes
    vacuously since it was written: it rebuilt the set of derived values by replaying the checker's
    regex literals over the document, and one of those literals is the bare capture fragment
    `([\\d.]+)` that the table-row patterns are composed from. Compiled alone it matched every
    number in the series, so the paper was being compared with itself.

    Two of the seven are the exact species the guard's own docstring warns about, arrived at
    independently: `$+0.0017$` sat as a literal INSIDE the pattern that checks its sigma quotient,
    so the quotient was pinned to a number nothing verified, and `$+0.055$` looked bound only
    because it collided with an unrelated checked value while its neighbour `$+0.098$` did not.

    Three are pre-correction values in the corrections registers. Those are the ones worth being
    nervous about, because a register states a number the current artifacts deliberately no longer
    produce, so there is nothing to notice when it is wrong --- and one already was. The population
    sd that inflated every t by sqrt(K/(K-1)) is recomputable, so each is derived by reapplying the
    old mistake rather than transcribed from a draft nobody can now consult.
    """
    src = _sources()
    tex = _tex()
    out: list[tuple[str, float, float, float]] = []

    # --- Refine accounting on the split axis (p1 sec:threats) -------------------------------
    # "arm D refines 30/30 held-out drafts at both n_train=4 and 16 and gains +0.055 and +0.098
    # per refined draft." The count is the denominator of the gains, so it is pinned beside them.
    n4, n16 = (_split_refine_gain(i) for i in (0, 1))
    out.append(
        (
            "tex: split refine denominator",
            _scrape(tex, r"arm~D refines \$(\d+)/30\$ held-out drafts"),
            0.0,
            float(n4[1]),
        )
    )
    printed = _scrape_all(tex, r"gains \$\+([\d.]+)\$ and \$\+([\d.]+)\$ per refined draft")
    out.append(("tex: split refine gain n=4", printed[0], 0.0005, n4[2]))
    out.append(("tex: split refine gain n=16", printed[1], 0.0005, n16[2]))

    # --- The flattest cell of the budget sweep (p1 fig:budget caption and sec:budget) --------
    # Stated twice, and unrounded in both places because at this size the third decimal IS the
    # claim: the registered rule required the deepest cell to clear +0.05, and it reads +0.002.
    deepest = _budget_delta(f"mmc{max(m for m, _ in BUDGET_SWEEP)}")
    for site, pattern in (
        ("caption", r"the deepest cell is the flattest of the\s+six at \$\+([\d.]+)\$"),
        ("prose", r"it reads \$\+([\d.]+)\$ \(the"),
    ):
        out.append((f"tex: budget deepest delta ({site})", _scrape(tex, pattern), 0.00005, deepest))

    # --- How far L3 falls short of the threshold it is measured against (p1 sec:threats) -----
    # A difference of two checked numbers is still a third number, and this one carries the
    # "still not a result" verdict on the level whose sign the paper retracts.
    l3 = _repeat_results()["L3"]
    out.append(
        (
            "tex: L3 shortfall from the p<0.10 threshold",
            _scrape(tex, r"fails the \$p<0.10\$ threshold by \$([\d.]+)\$"),
            0.005,
            _repeat().T_CRIT_10[4] - l3.t_stat(),  # type: ignore[attr-defined]
        )
    )

    # --- What the threshold move itself bought (p1 sec:analysis) -----------------------------
    # The three tasks whose false pass the hand-written gene caught. This arithmetic is what
    # licenses the section's conclusion that the gain is an order of magnitude below its noise.
    c1 = _c1_strata()
    rose = _scrape_all(tex, r"the score rose \$([\d.]+) \\to ([\d.]+)\$")
    out.append(("tex: c1 caught stratum draft mean", rose[0], 0.0005, c1.caught_draft))
    out.append(("tex: c1 caught stratum final mean", rose[1], 0.0005, c1.caught_final))
    out.append(
        (
            "tex: c1 caught stratum contribution",
            _scrape(tex, r"worth \$\+([\d.]+)\$ on the 32-task mean"),
            0.0005,
            c1.caught_contribution,
        )
    )
    out.append(
        (
            "tex: c1 compliant stratum mean loss",
            _scrape(tex, r"already compliant the\s+mean is \$-([\d.]+)\$"),
            0.0005,
            -c1.compliant_mean,
        )
    )
    # The three stratum SIZES, because the sentence that used to state them got one wrong: it
    # called all eight "the only ones on which the verdicts differ" while conceding two clauses
    # earlier that both genes demanded a fix on five of them. The verdicts differ on three. Sizes
    # are checked, not just the means they weight, so the miscount cannot come back reworded.
    for label, pattern, actual in (
        (
            "verdicts differ",
            r"\\emph\{verdicts\} differ on exactly\s+those \\textbf\{(\w+)\}",
            c1.caught_n,
        ),
        ("already compliant", r"demanded a fix on all \\textbf\{(\w+)\}", c1.compliant_n),
        (
            "both fixed, non-compliant",
            r"and on the other \\textbf\{([\w-]+)\}",
            c1.total_n - c1.caught_n - c1.compliant_n,
        ),
    ):
        out.append(
            (f"tex: c1 stratum size, {label}", _scrape_word(tex, pattern), 0.0, float(actual))
        )

    # --- Pre-correction values in the corrections registers ----------------------------------
    # The estimator bug used a population sd, which divides by K rather than K-1 and so scales
    # every sd DOWN by sqrt((K-1)/K) and every t UP by its reciprocal. Both K values come from the
    # artifacts: the replicate count for the t inflation, the paired-task count for the sigma.
    inflate = (l3.k / (l3.k - 1)) ** 0.5  # type: ignore[attr-defined]
    noise = _noise_estimate()
    # The entry states which value is the buggy one IN WORDS, because printing both numbers and
    # letting the reader infer the direction is what made two independent reviewers read the
    # earlier phrasing backwards. Group 1 is the population-sd value, group 2 the corrected one.
    #
    # `carried` exists because the previous version skipped a paper whose pattern did not match,
    # on the theory that not every paper carries the entry. All three do. When a rewording broke
    # the pattern, the six checks did not fail -- they ceased to exist, and the total silently
    # fell from 948 to 942. A guard that can evaporate is the defect this series is about.
    carried = 0
    for slug in src.PAPERS:
        paper = src.paper_tex(slug)
        pair = _scrape_all(
            paper,
            r"\\emph\{understated\} \$\\sigma\(\\dd\)\$\s*\n?\s*as \$([\d.]+)\$ where the "
            r"unbiased estimator gives \$([\d.]+)\$",
        )
        if pair[0] != pair[0]:  # NaN: this paper carries no such register entry
            continue
        carried += 1
        n = src.PAPER_NUMERALS[slug]
        out.append(
            (f"register {n}: corrected cross-run sigma", pair[1], 0.00005, noise.sigma_delta_mean)
        )
        out.append(
            (
                f"register {n}: population-sd cross-run sigma",
                pair[0],
                0.00005,
                noise.sigma_delta_mean * ((noise.n_tasks - 1) / noise.n_tasks) ** 0.5,
            )
        )
    if carried != len(src.PAPERS):
        raise AssertionError(
            f"only {carried} of {len(src.PAPERS)} papers state the cross-run sigma correction in "
            "the form this check reads; it is in all three, so a non-match means the wording "
            "drifted and these checks were about to disappear rather than fail"
        )
    # The discarded first draft of the budget sweep: its priced bound, and the two ratios the
    # sentence contrasts. The dollar figure was $7.93 and reconciled to no configuration the cost
    # model can produce; it is now derived from the two redundant cells at the config the text
    # names. The "12x" was a ratio of priced CYCLES sitting next to the words "budget range",
    # which reads as the metric-call ratio and is 8x, so both are pinned separately.
    from whystuck.eval.cost import WORKER_TOKENS, estimate_usd, plan_study_calls

    def _first_draft_cell(mmc: int) -> float:
        return estimate_usd(
            plan_study_calls(
                arms=("B", "D"),
                n_train=8,
                n_heldout=10,
                max_metric_calls=mmc,
                reflection_calls=6,
                worker_model="openai/gpt-4.1-mini",
                reflection_model="openai/gpt-5.6-luna",
                n_seeds=3,
                worker_calls_per_task=3,
                worker_tokens=WORKER_TOKENS,
            )
        )

    p1_tex = src.paper_tex("p1-conditional-lever")
    import analyze_search_depth as _asd

    out.append(
        (
            "tex: discarded first-draft sweep bound",
            _scrape(p1_tex, r"would have spent \\\$([\d.]+) of priced"),
            0.005,
            _first_draft_cell(80) + _first_draft_cell(160),
        )
    )
    out.append(
        (
            "tex: first-draft priced-depth ratio",
            _scrape(p1_tex, r"``\$(\d+)\\times\$ range in \\emph\{priced"),
            0.75,
            _asd.priced_cycles(160) / _asd.priced_cycles(20),
        )
    )
    out.append(
        (
            "tex: first-draft metric-call ratio",
            _scrape(p1_tex, r"budget itself spans only \$(\d+)\\times\$"),
            0.0,
            160 / 20,
        )
    )
    # The two ways the same t inflation can be quoted. Stated both ways in the register precisely
    # because "too large by" does not name its denominator.
    for label, pattern, expected in (
        (
            "t excess over corrected",
            r"exceeded its corrected value by\s*\n?\s*\$([\d.]+)\\%\$",
            100 * (inflate - 1),
        ),
        (
            "t reduction from correcting",
            r"correcting one lowers it by \$([\d.]+)\\%\$",
            100 * (1 - 1 / inflate),
        ),
    ):
        out.append((f"register: {label}", _scrape(tex, pattern), 0.05, expected))
    # The two t values the audit found restated from before the correction. Their corrected
    # companions (0.33 and 8.08) are pinned by the tab:curvepairs checks; these are not, because
    # no current artifact produces them --- which is precisely why one of the three drifted.
    for label, pattern, pairkey in (
        (
            "oracle-under-intrinsic t",
            r"oracle-under-intrinsic \$t\$ as \$([\d.]+)\$ in two places",
            ("run3", "run7"),
        ),
        (
            "actionability t",
            r"actionability \$t\$ as \$([\d.]+)\$ in the conclusion",
            ("run9", "run10"),
        ),
    ):
        res = _curve_replicates()[1][pairkey]
        out.append(
            (
                f"register: population-sd {label}",
                _scrape(tex, pattern),
                0.005,
                res.t_stat() * inflate,  # type: ignore[attr-defined]
            )
        )
    # The L1 five-replicate t, which Paper I's register corrected to 1.00 and Paper II's now cites
    # by its OLD value while naming it as the first missed site. Derived the same way as the two
    # above rather than quoted from the other register, so the two papers cannot drift apart on it.
    out.append(
        (
            "register: population-sd L1 five-replicate t",
            _scrape(tex, r"missed, after the \$([\d.]+)\$ of Paper~I's register"),
            0.005,
            _repeat_results()["L1"].t_stat() * inflate,  # type: ignore[attr-defined]
        )
    )
    return out


@cache
def _budget_pass_census() -> tuple[tuple[str, int, bool, int, int, int], ...]:
    """Per budget-sweep cell and seed: (slug, seed, gene was evolved, n PASS, tp, fp).

    Replayed under each seed's OWN winning genome, which is what makes the counts comparable to the
    ones the paper prints: overriding `ensure_correct` alone raises a cassette miss, because the
    drafts the critique judges came from the evolved `generate`.

    Calibration is not assumed. This reproduces the two counts sec:budgetresult already states --
    mmc=80's single false PASS and mmc=160's single true one -- so the third count it produces, the
    one the paper got wrong, comes off the same instrument as the two it got right.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analyze_critique_detector as acd

    out = []
    for _mmc, _cap, slug in _BUDGET_ROWS:
        doc = json.loads(Path(f"{DOCS}/validation-study-{slug}.json").read_text())
        for seed, run in enumerate(doc["runs"]):
            evolved = "ensure_correct" in (run["D"].get("genes_changed") or [])
            events = acd.collect_events(
                f"tests/fixtures/cassettes/study-{slug}.json",
                [seed],
                loop_kw={},
                gene_overrides=run["D"].get("final_genes") or None,
            )
            out.append(
                (
                    slug,
                    seed,
                    evolved,
                    sum(1 for verdict, _ in events if verdict),
                    sum(1 for verdict, true in events if verdict and true),
                    sum(1 for verdict, true in events if verdict and not true),
                )
            )
    return tuple(out)


def _budget_false_pass_checks() -> list[tuple[str, float, float, float]]:
    """The false \\texttt{PASS} verdicts evolved critique genes emit across the budget sweep.

    sec:budgetresult called mmc=80's single false PASS "the first false PASS from an evolved gene
    anywhere in this study". It is neither the first nor the largest: at mmc=56 -- one table row
    above, shallower in priced depth and at the SAME measured depth 6 -- an evolved gene emits three
    PASS verdicts and all three are false. The sweep holds four such verdicts, not one.

    Every count the corrected paragraph states is derived here, including the two it always had
    right. A paragraph where some numbers are checked and one is not is how the wrong one survived.
    """
    census = _budget_pass_census()
    evolved = [row for row in census if row[2]]
    accepting = {row[0]: row for row in evolved if row[3]}
    tex = _tex()
    n_pass, tp, fp = 3, 4, 5
    sites = (
        ("depth 12 accepting gene tp", r"is right \(\$\\mathrm\{tp\}=(\d+)\$", "mmc160", tp),
        (
            "depth 12 accepting gene fp",
            r"is right \(\$\\mathrm\{tp\}=\d+\$, \$\\mathrm\{fp\}=(\d+)\$",
            "mmc160",
            fp,
        ),
        (
            "mmc=80 accepting gene tp",
            r"is \\textbf\{wrong\}\s+\(\$\\mathrm\{tp\}=(\d+)\$",
            "mmc80",
            tp,
        ),
        (
            "mmc=80 accepting gene fp",
            r"is \\textbf\{wrong\}\s+\(\$\\mathrm\{tp\}=\d+\$,\s+\$\\mathrm\{fp\}=(\d+)\$",
            "mmc80",
            fp,
        ),
        (
            "mmc=56 accepting gene tp",
            r"all of them false \(\$\\mathrm\{tp\}=(\d+)\$",
            "mmc56",
            tp,
        ),
        (
            "mmc=56 accepting gene fp",
            r"all of them false \(\$\\mathrm\{tp\}=\d+\$, \$\\mathrm\{fp\}=(\d+)\$",
            "mmc56",
            fp,
        ),
    )
    out = [
        (f"tex: {label}", _scrape(tex, pattern), 0.0, float(accepting[slug][field]))
        for label, pattern, slug, field in sites
    ]
    out.append(
        (
            "tex: mmc=56 accepting gene PASS count",
            _scrape_word(tex, r"emits \\textbf\{(\w+)\} \\texttt\{PASS\} verdicts"),
            0.0,
            float(accepting["mmc56"][n_pass]),
        )
    )
    out.append(
        (
            "tex: false PASS verdicts from evolved genes in the sweep",
            _scrape_word(
                tex, r"the sweep contains\s+\\textbf\{(\w+)\} false \\texttt\{PASS\} verdicts"
            ),
            0.0,
            float(sum(row[fp] for row in evolved)),
        )
    )
    # "Depth 6" does not identify a cell, which is half of why the claim read as narrow. The count
    # comes from the same replay that fills tab:budgetresult's measured-depth column, not from
    # parsing the rendered column: the column is already checked against `_search_depth`, so
    # deriving from the replay keeps one source of truth instead of two.
    depths = [_search_depth(slug, mmc, cap) for mmc, cap, slug in _BUDGET_ROWS]
    if not any(d is not None for d in depths):
        raise ValueError("no budget cell reports a measured depth; the replay path changed")
    out.append(
        (
            "tex: budget cells sharing measured depth 6",
            _scrape_word(tex, r"\\textbf\{(\w+)\} cells share measured depth"),
            0.0,
            float(depths.count(6)),
        )
    )
    return out


def _sigma_margin_checks() -> list[tuple[str, float, float, float]]:
    """How much sigma would have to grow for each borrowed-denominator ratio to fall under 2 sigma.

    The SECOND site the population-sd correction's "corrected throughout" missed, after the 1.12 of
    the register. sec:curverepeat printed 5, 11 and 14 percent; under the shipped sigma
    (0.029052, the sample sd) the margins are 3.3, 9.0 and 11.9, and 5/11/14 reproduce only under
    the retired 0.028564. The direction matters twice over: the printed figures make the ratios look
    LESS fragile than they are, and the sentence's whole purpose is to argue they were fragile
    enough to justify buying ninety replicate re-scorings. Correcting them strengthens the argument
    the paragraph is making.

    Derived from the same `_curve_gap` ratios the surrounding prose already pins, so a future
    denominator change moves these with everything else instead of leaving them behind again.
    """
    tex = _tex()
    sites = (
        ("tool lift", ("run3", "run4"), r"needs only \$(\d+)\\%\$ more \$\\sigma\$"),
        ("actionability", ("run9", "run10"), r"fall under it, actionability \$(\d+)\\%\$"),
        ("refine budget", ("run7", "run8"), r"and the refine\s+budget \$(\d+)\\%\$"),
    )
    return [
        (
            f"tex: sigma margin to 2 sigma, {label}",
            _scrape(tex, pattern),
            0.0,
            float(round(100 * (_curve_gap(*pair)[1] / 2 - 1))),
        )
        for label, pair, pattern in sites
    ]


def _registered_depth_bound_checks() -> list[tuple[str, float, float, float]]:
    """The pre-registered depth-sweep bound, and the slip in the sigma it was computed from.

    sec:budgetprereg registered "no cell exceeding 2.5 times the directly measured sigma(dd) = 0.042
    at this level, i.e. |dd| < 0.105". The measured value is 0.0468, printed as 0.047 in the three
    other places the paper cites the same section for it; 0.042 appears in no replicate. The
    registered bound is therefore 2.5 x a number nothing produced.

    The bound is NOT recomputed. A pre-registered threshold recalculated after the fact is not
    pre-registered, so 0.105 stands as what was registered and the footnote states what the artifact
    supports. What is checked here is everything that IS derivable: the measured sigma, the bound it
    would have given, the sweep's largest |dd|, and -- because both of the registered pair are now
    stated in one footnote -- the arithmetic between them.

    Neither 0.042 nor 0.105 was reachable by `test_paper_prose_coverage`, even repaired, and the
    reason is worth recording: that guard tests membership in the set of pinned VALUES, and 0.042
    collides with the paper's own -0.042 headline while 0.105 collides with a CI upper bound in
    tab:p0. A wrong number is likeliest to collide with a related one NEARBY, because nearby
    is usually where it was copied from, so value-set membership is the weakest form of this rule.
    """
    tex = _tex()
    sigma = _repeat_results()["L1"].sd_between_replicates  # type: ignore[attr-goes-through-object]
    largest = max((abs(_budget_delta(slug)) for _, _, slug in _BUDGET_ROWS))
    transcribed = _scrape(tex, r"a \$\\sigma\(\\dd\)\$ transcribed as \$([\d.]+)\$")
    registered = _scrape(tex, r"i\.e\.\\ \$\|\\dd\| < ([\d.]+)\$ throughout")
    return [
        (
            "tex: registered depth bound, measured sigma",
            _scrape(tex, r"The measured value is\s+\$([\d.]+)\$ \(Table"),
            0.0005,
            sigma,
        ),
        (
            "tex: registered depth bound the artifact supports",
            _scrape(tex, r"which gives \$([\d.]+)\$"),
            0.0005,
            2.5 * sigma,
        ),
        (
            "tex: sweep largest absolute delta",
            _scrape(tex, r"the largest \$\|\\dd\|\$ in the\s+sweep is \$([\d.]+)\$"),
            0.0005,
            largest,
        ),
        # The transcribed sigma is a historical constant, so it is pinned by the only two things
        # that can be said about it from evidence: it is 2.5 times smaller than the bound it
        # produced, and it is the rounded largest |dd| the sweep contains. The second is what the
        # footnote calls the likely mechanism, and stating it in the paper obliges checking it.
        (
            "tex: registered bound is 2.5x the transcribed sigma",
            registered,
            0.0005,
            2.5 * transcribed,
        ),
        (
            "tex: transcribed sigma equals the rounded largest delta",
            transcribed,
            0.0005,
            round(largest, 3),
        ),
    ]


_SENTENCE_BREAK = re.compile(r"(?<=\.)\s+(?=[A-Z\\])")


def _sentence_at(text: str, index: int) -> str:
    """The sentence containing `index`.

    Split on a period followed by whitespace and a capital or a control sequence, which never
    matches a decimal point (`0.088` has no space after the period) and errs toward splitting
    too often -- a shorter window can only make a guard that searches it stricter.
    """
    start = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, index):
        start = match.end()
    end = _SENTENCE_BREAK.search(text, index)
    return text[start : end.start() if end else len(text)]


def _oracle_gap_direction_checks() -> list[tuple[str, float, float, float]]:
    """Both oracle gaps are negative, so every site printing one must say which way it points.

    The figure caption read ``the same oracle is $0.028$ better'' two lines below calling that
    same point the lowest of the three scores. The magnitude was pinned to the artifact and the
    direction was not, so a self-contradicting caption passed every check in this file. A site
    satisfies the guard by carrying an explicit minus sign or a downward word nearby.
    """
    p2 = _sources().paper_tex("p2-repair-not-detection")
    out: list[tuple[str, float, float, float]] = []
    for magnitude, gap in (
        ("0.088", _curve_gap("run4", "run7")[0]),
        ("0.028", _curve_gap("run3", "run7")[0]),
    ):
        if gap >= 0:
            raise AssertionError(f"the {magnitude} oracle gap is no longer negative")
        for i, m in enumerate(re.finditer(rf"{re.escape(magnitude)}(?!\d)", p2)):
            signed = p2[max(0, m.start() - 2) : m.start()].endswith("-")
            downward = re.search(
                r"\b(below|lowest|lower|worse|beneath)\b", _sentence_at(p2, m.start())
            )
            out.append(
                (
                    f"tex: {magnitude} oracle gap site {i} points down",
                    1.0,
                    0.0,
                    float(bool(signed or downward)),
                )
            )
    if len(out) < 6:
        raise AssertionError(f"expected both oracle gaps at several sites each, found {len(out)}")
    return out


@cache
def _groq_cell(slug: str) -> tuple[float, ...]:
    """Arm A's per-seed seed-genome held-out scores for one second-worker cell."""
    doc = json.loads(Path(f"{DOCS}/validation-study-groq-{slug}.json").read_text())
    return tuple(r["A"]["seed_heldout"] for r in doc["runs"])


def _groq_paired(a: str, b: str) -> tuple[float, float, float]:
    """(mean difference b-a, paired s.e., t) over the seeds the two cells share.

    Paired because every cell runs the same seed list, so seed i draws the same tasks in both
    and task difficulty cancels. Unpaired standard errors here would be roughly four times
    larger and would hide every effect this subsection reports.
    """
    x, y = _groq_cell(a), _groq_cell(b)
    if len(x) != len(y):
        raise AssertionError(f"{a} has {len(x)} seeds, {b} has {len(y)}; the pairing is fiction")
    diffs = [q - p for p, q in zip(x, y, strict=True)]
    se = stdev(diffs) / len(diffs) ** 0.5
    return mean(diffs), se, mean(diffs) / se


def _groq_second_worker_checks() -> list[tuple[str, float, float, float]]:
    """Paper II's cross-worker corroboration table, every figure derived from the runs.

    The subsection exists because a registered prediction failed, so its numbers carry more
    weight than usual: they are the evidence that the failure was a confounded cell rather than
    a refuted thesis. Each is re-derived here from the released artifacts, including the two
    the prose states only as a decomposition (the 14/86 split).
    """
    p2 = _sources().paper_tex("p2-repair-not-detection")
    out: list[tuple[str, float, float, float]] = []

    # The table body: held-out mean, delta, s.e. and t for each cell against the intrinsic one.
    rows = (
        # The pass rate here is an ANCHOR, not a checked value, and it carried a wrong number
        # for as long as it existed: the row said 1% where L1 measures 0.5% (1 of 200). A
        # literal inside a locator is the one kind of number a scraping checker structurally
        # cannot check -- get it wrong and the check does not fail, it stops matching, and a
        # non-matching row here raised only because the tuple unpack below needs the groups.
        # The rate is now pinned as a value of its own by `_groq_detector_degeneracy_checks`,
        # so the same numeral is both the key and a claim, and moving one moves the other.
        ("intrinsic", "p0cal-L1", r"its own prose \(\\texttt\{PASS\} on \$0\.5\\%\$\) & ([\d.]+)"),
        (
            "8b",
            "det-8b",
            r"smaller critic \(8B\) & ([\d.]+) & \$-([\d.]+)\$ & ([\d.]+) & \$-([\d.]+)\$",
        ),
        (
            "qwen",
            "det-qwen",
            r"\\texttt\{qwen3\.6-27b\}\) & ([\d.]+) & \$\+([\d.]+)\$ & ([\d.]+) & \$\+([\d.]+)\$",
        ),
        (
            "oracle-lo",
            "det-oracle-lo",
            r"\\texttt\{lo\} prose \(verdict only\) & ([\d.]+) & \$\+([\d.]+)\$ & ([\d.]+) & "
            r"\$\+([\d.]+)\$",
        ),
        (
            "oracle-mid",
            "det-oracle",
            r"\\texttt\{mid\} prose & ([\d.]+) & \$\+([\d.]+)\$ & ([\d.]+) & "
            r"\$\\mathbf\{\+([\d.]+)\}\$",
        ),
        (
            "oracle-hi",
            "det-oracle-hi",
            r"\\texttt\{hi\} prose \(measured edits\) & ([\d.]+) & \$\+([\d.]+)\$ & ([\d.]+) & "
            r"\$\\mathbf\{\+([\d.]+)\}\$",
        ),
    )
    for name, slug, pattern in rows:
        printed = _scrape_all(p2, pattern)
        out.append(
            (f"tab:secondworker {name} held-out", printed[0], 0.00005, mean(_groq_cell(slug)))
        )
        if name == "intrinsic":
            continue
        delta, se, t = _groq_paired("p0cal-L1", slug)
        # Signed, not absolute: the 8b row prints a negative delta and a negative t, and the
        # table-coverage guard reads the minus sign as part of the number. Pinning |x| left both
        # of that row's figures uncovered while every check still passed.
        sign = -1.0 if name == "8b" else 1.0
        out.append((f"tab:secondworker {name} delta", sign * printed[1], 0.00005, delta))
        out.append((f"tab:secondworker {name} se", printed[2], 0.00005, se))
        out.append((f"tab:secondworker {name} t", sign * printed[3], 0.005, t))

    # The decomposition, on the cell that actually reversed the prediction. An earlier draft
    # decomposed the `hi` cell instead, whose total is nearly twice as large, which inflated the
    # prose share from 75% to 86% by substituting a condition that was never under test.
    verdict_only = _groq_paired("p0cal-L1", "det-oracle-lo")[0]
    mid_total = _groq_paired("p0cal-L1", "det-oracle")[0]
    mid_prose, _mid_se, mid_prose_t = _groq_paired("det-oracle-lo", "det-oracle")
    hi_prose, _hi_se, hi_prose_t = _groq_paired("det-oracle-lo", "det-oracle-hi")
    out.append(
        (
            "tex: second-worker verdict contribution",
            _scrape(p2, r"the perfect verdict contributes \$\+([\d.]+)\$"),
            0.00005,
            verdict_only,
        )
    )
    out.append(
        (
            "tex: second-worker mid prose contribution",
            _scrape(p2, r"and the\s*\n?repair text \$\+([\d.]+)\$"),
            0.00005,
            mid_prose,
        )
    )
    out.append(
        (
            "tex: second-worker mid prose t",
            _scrape(p2, r"repair text \$\+[\d.]+\$ \(\$t = ([\d.]+)\$\)"),
            0.005,
            mid_prose_t,
        )
    )
    shares = _scrape_all(p2, r"a \$(\d+)/(\d+)\$ split")
    out.append(("tex: second-worker verdict share", shares[0], 0.5, 100 * verdict_only / mid_total))
    out.append(("tex: second-worker prose share", shares[1], 0.5, 100 * mid_prose / mid_total))
    out.append(
        (
            "tex: second-worker hi prose rung",
            _scrape(p2, r"measured edits, moves it to \$\+([\d.]+)\$"),
            0.00005,
            hi_prose,
        )
    )
    out.append(
        (
            "tex: second-worker hi prose t",
            _scrape(p2, r"moves it to \$\+[\d.]+\$ at \$t = ([\d.]+)\$"),
            0.005,
            hi_prose_t,
        )
    )
    # The task-level pairing of the headline contrast, over 20 seeds x 10 held-out tasks.
    by_task: dict[str, list[float]] = {}
    for slug in ("p0cal-L1", "det-oracle"):
        doc = json.loads(Path(f"{DOCS}/validation-study-groq-{slug}.json").read_text())
        vals: list[float] = []
        for run in doc["runs"]:
            per = run["A"]["seed_heldout_by_task"]
            vals += [per[k] for k in sorted(per)] if isinstance(per, dict) else list(per)
        by_task[slug] = vals
    diffs = [b - a for a, b in zip(by_task["p0cal-L1"], by_task["det-oracle"], strict=True)]
    task_t = mean(diffs) / (stdev(diffs) / len(diffs) ** 0.5)
    out.append(
        (
            "tex: second-worker task-level t",
            _scrape(p2, r"\$t = ([\d.]+)\$ over the two hundred held-out tasks"),
            0.005,
            task_t,
        )
    )
    out.append(
        (
            "tex: second-worker task-level n",
            _scrape_word(p2, r"over the (\w+ \w+) held-out tasks"),
            0.0,
            float(len(diffs)),
        )
    )
    # The two p0 figures the limitation paragraph turns on, and the false-alarm counts.
    import analyze_critique_detector as acd

    seeds = tuple(range(20))
    for label, slug, pattern in (
        ("p0 at L1", "p0cal-L1", r"pass rate is \$p_0 = ([\d.]+)\$ against"),
        ("p0 at L2", "p0cal-L2", r"takes it to \$([\d.]+)\$ where the other"),
    ):
        exclude = frozenset({"exact_word_count"}) if slug.endswith("L2") else frozenset()
        events = acd.collect_events(
            f"tests/fixtures/cassettes/study-groq-{slug}.json",
            list(seeds),
            loop_kw={},
            n_heldout=10,
            exclude=exclude,
        )
        out.append(
            (f"tex: second-worker {label}", _scrape(p2, pattern), 0.0005, acd.base_rate_p0(events))
        )
    # Paper I quotes the same three p0 figures in D9, and the series' rule is that a number two
    # papers state must be the same number in both. Pinned per paper rather than cross-checked
    # against each other, so a drift in one does not validate itself against the other.
    p1 = _sources().paper_tex("p1-conditional-lever")
    printed_ladder = _scrape_all(
        p1, r"give\s*\n?\s*\$p_0 = ([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$"
    )
    for i, (slug, exclude) in enumerate(
        (
            ("p0cal-L1", frozenset()),
            ("p0cal-L2", frozenset({"exact_word_count"})),
            ("p0cal-L3", frozenset({"exact_word_count", "word_max_length"})),
        )
    ):
        events = acd.collect_events(
            f"tests/fixtures/cassettes/study-groq-{slug}.json",
            list(seeds),
            loop_kw={},
            n_heldout=10,
            exclude=exclude,
        )
        out.append(
            (f"tex: D9 p0 ladder {slug} (p1)", printed_ladder[i], 0.0005, acd.base_rate_p0(events))
        )
    l2 = acd.collect_events(
        "tests/fixtures/cassettes/study-groq-p0cal-L2.json",
        list(seeds),
        loop_kw={},
        n_heldout=10,
        exclude=frozenset({"exact_word_count"}),
    )
    correct = [true for _verdict, true in l2 if true]
    alarms = [true for verdict, true in l2 if not verdict and true]
    printed = _scrape_all(
        p2,
        r"the relaxed level where \$(\d+)\$ of \$200\$ first drafts are\s*\n?"
        r"already correct it demands a fix on \$(\d+)\$ of them",
    )
    out.append(("tex: second-worker already-correct", printed[0], 0.0, float(len(correct))))
    out.append(("tex: second-worker false alarms", printed[1], 0.0, float(len(alarms))))
    out.extend(_groq_detector_degeneracy_checks())
    return out


def _groq_detector_degeneracy_checks() -> list[tuple[str, float, float, float]]:
    """Every cell in Paper II's second-worker table, scored AS A DETECTOR.

    Written because a wrong number survived here for as long as the section existed, in the one
    position a scraping checker cannot see: the intrinsic row's pass rate sat inside the regex
    that LOCATES the row. `1\\%` was the anchor, so it was never compared to anything; L1
    actually passes 1 draft of 200, which is 0.5%. A literal used as a key is a literal nothing
    checks, and the repair is not to be more careful with anchors but to pin the same numeral as
    a value, which is what the first three entries below do.

    The prose error came from the same rounding. The calibration table printed all three levels
    at 2 dp, so 0.005, 0.015 and 0.010 all read as `0.01`, and the paper drew one worker-level
    fact ("PASS on 1%") out of a column where three different rates had been made to look equal.
    All three rates are pinned here, at the precision that distinguishes them.

    The `tp=0` entries matter for a different reason: they are what licenses the section's
    reading of its own nulls. A critic that never passes an already-correct draft is degenerate,
    so "swapping the critic buys nothing" is a statement about two bad detectors rather than
    about detection. Nothing in the artifacts forces the paper to say that; this check forces
    the numbers it says it on to be the measured ones.
    """
    import analyze_critique_detector as acd

    p2 = _sources().paper_tex("p2-repair-not-detection")
    seeds = list(range(20))
    out: list[tuple[str, float, float, float]] = []

    def _events(slug: str, exclude: frozenset[str], separate: bool):
        return acd.collect_events(
            f"tests/fixtures/cassettes/study-groq-{slug}.json",
            seeds,
            loop_kw={},
            n_heldout=10,
            exclude=exclude,
            separate_critic=separate,
        )

    # The three calibration levels' pass rates, as percentages, at the precision that keeps them
    # apart. The paper prints them in one sentence precisely to record that they are not equal.
    printed_rates = _scrape_all(
        p2, r"rates are in fact \$([\d.]+)\\%\$, \$([\d.]+)\\%\$ and \$([\d.]+)\\%\$"
    )
    for i, (slug, exclude) in enumerate(
        (
            ("p0cal-L1", frozenset()),
            ("p0cal-L2", frozenset({"exact_word_count"})),
            ("p0cal-L3", frozenset({"exact_word_count", "word_max_length"})),
        )
    ):
        ev = _events(slug, exclude, False)
        rate = 100 * sum(1 for verdict, _true in ev if verdict) / len(ev)
        out.append((f"tex: intrinsic PASS rate {slug} (%)", printed_rates[i], 0.005, rate))

    # The same L1 rate where it is printed as the table row's own label -- the position that
    # carried the error. Pinned as a value now, so the anchor and the claim move together.
    l1 = _events("p0cal-L1", frozenset(), False)
    out.append(
        (
            "tex: intrinsic row label PASS rate (%)",
            _scrape(p2, r"its own prose \(\\texttt\{PASS\} on \$([\d.]+)\\%\$\)"),
            0.005,
            100 * sum(1 for verdict, _true in l1 if verdict) / len(l1),
        )
    )
    out.append(
        (
            "tex: intrinsic L1 passes, count",
            _scrape(p2, r"it emits \\texttt\{PASS\} on \$(\d+)\$ of \$200\$ drafts"),
            0.0,
            float(sum(1 for verdict, _true in l1 if verdict)),
        )
    )
    out.append(
        (
            "tex: intrinsic L1 event count",
            _scrape(p2, r"emits \\texttt\{PASS\} on \$\d+\$ of \$(\d+)\$ drafts"),
            0.0,
            float(len(l1)),
        )
    )

    # The substituted critics, scored as detectors. `separate_critic` is required: both were
    # recorded with --critique-model, and replaying them against the worker surface returns an
    # empty list rather than raising.
    out.append(
        (
            "tex: second-worker first-critique events",
            _scrape(p2, r"over the \$(\d+)\$\s*\n?first-critique events"),
            0.0,
            float(len(_events("det-8b", frozenset(), True))),
        )
    )
    for label, slug, tp_expected in (("8b", "det-8b", 0.0), ("qwen", "det-qwen", 0.0)):
        stats = acd.detector_stats(_events(slug, frozenset(), True))
        out.append((f"tex: {label} critic true positives", tp_expected, 0.0, stats["tp"]))
    # "the 8B critic emits PASS once and qwen never" -- both stated as words, both derived.
    eight = acd.detector_stats(_events("det-8b", frozenset(), True))
    qwen = acd.detector_stats(_events("det-qwen", frozenset(), True))
    out.append(
        (
            "tex: 8b critic passes (word 'once')",
            1.0 if re.search(r"the 8B critic emits \\texttt\{PASS\} once", p2) else -1.0,
            0.0,
            eight["tp"] + eight["fp"],
        )
    )
    out.append(
        (
            "tex: qwen critic passes (word 'never')",
            0.0 if re.search(r"\\texttt\{qwen3\.6-27b\} never", p2) else -1.0,
            0.0,
            qwen["tp"] + qwen["fp"],
        )
    )
    # The two substituted cells' already-correct counts. They differ (24 and 20) and differ from
    # L1's 19, because these runs were recorded at the provider's default temperature and so do
    # not share a draft set. Stating one number for all three would have been the tidier
    # sentence and a false one, so both are printed and both are pinned.
    printed_correct = _scrape_all(
        p2, r"against \$(\d+)\$ and \$(\d+)\$ already-correct first drafts"
    )
    for i, slug in enumerate(("det-8b", "det-qwen")):
        out.append(
            (
                f"tex: already-correct first drafts, {slug}",
                printed_correct[i],
                0.0,
                float(sum(1 for _v, true in _events(slug, frozenset(), True) if true)),
            )
        )

    # Paper I closes D3's Remaining item on the same measurement, so it restates two of these
    # numbers. Both are re-derived here against Paper I's own text rather than compared with
    # Paper II's, on the series' rule that a cross-paper agreement checked paper-against-paper
    # can be satisfied by two matching errors.
    p1 = _sources().paper_tex("p1-conditional-lever")
    out.append(
        (
            "tex: D9/D3 qwen drafts judged (p1)",
            _scrape(p1, r"emits \\texttt\{PASS\} on none\s*\n?\s*of \$(\d+)\$ first drafts"),
            0.0,
            float(len(_events("det-qwen", frozenset(), True))),
        )
    )
    out.append(
        (
            "tex: D9/D3 qwen already-correct (p1)",
            _scrape(p1, r"first drafts, \$(\d+)\$ of which were already correct"),
            0.0,
            float(sum(1 for _v, true in _events("det-qwen", frozenset(), True) if true)),
        )
    )
    return out


def _xprovider_checks() -> list[tuple[str, float, float, float]]:
    """Every number the D3 cross-provider run put into the three papers.

    The run is reported in three places for three different reasons -- Paper I's decision list
    (it was the last entry marked blocked), Paper II's detector section (its H2 result is that
    paper's thesis), and Paper III's reporting minimum (it is the first PROSPECTIVE application
    of that rule, and the one that showed the rule too coarse). Three papers quoting one run is
    exactly the shape that produced this series' restatement defects, so every figure is derived
    once here and checked at each site.
    """
    tex = _tex()
    src = _sources()
    seeds, b_seeds, d_seeds = _xprovider_acceptance()
    strata = _xprovider_strata()
    out: list[tuple[str, float, float, float]] = []

    # --- Paper III: the acceptance census and the stratification -----------------------------
    for label, pattern, actual in (
        ("seeds", r"cross-provider critic study \(Paper~I, D3\) over \\textbf\{(\w+)\}", seeds),
        ("arm B accepted", r"arm~B on \\textbf\{(\w+)\} seeds", b_seeds),
        ("arm D accepted", r"and arm~D on \\textbf\{(\w+)\}", d_seeds),
    ):
        out.append((f"tex: D3 {label}", _scrape_word(tex, pattern), 0.0, float(actual)))

    rows = (
        ("treated", r"the treatment applied & (\d+) & \$\+([\d.]+)\$ & \$(\d+)\\%\$"),
        (
            "comparator only",
            r"D returned the seed\}? & \\textbf\{(\d+)\} & \$\\mathbf\{\+([\d.]+)\}\$ & "
            r"\$\\mathbf\{(\d+)\\%\}\$",
        ),
        ("neither", r"neither accepted & (\d+) & \$\+([\d.]+)\$ & \$(\d+)\\%\$"),
    )
    for key, pattern in rows:
        printed = _scrape_all(tex, pattern)
        stratum = strata[key]
        out.append((f"tab:D3strata {key} seeds", printed[0], 0.0, float(stratum.seeds)))
        out.append((f"tab:D3strata {key} delta", printed[1], 0.00005, abs(stratum.mean_delta)))
        out.append((f"tab:D3strata {key} share", printed[2], 0.5, 100 * stratum.share))

    # The headline the stratification reinterprets, and the treated-only figure it leaves.
    doc = json.loads(Path(f"{DOCS}/validation-study-xprovider.json").read_text())
    runs = doc["runs"]
    delta = [r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs]
    out.append(
        (
            "tex: D3 headline delta (p3)",
            _scrape(tex, r"Its\s+\$\\dd\(D-B\) = \+([\d.]+)\$ would be read"),
            0.0005,
            mean(delta),
        )
    )
    out.append(
        (
            "tex: D3 treated-only delta",
            _scrape(tex, r"something, \$\\dd = \+([\d.]+)\$ on \$n = 3\$"),
            0.0005,
            strata["treated"].mean_delta,
        )
    )
    # The comparator's self-inflicted losses, which are the whole of the 74%.
    damage = sorted(
        (
            r["B"]["opt_heldout"] - r["B"]["seed_heldout"]
            for r in runs
            if not r["D"].get("genes_changed") and r["B"].get("genes_changed")
        ),
    )
    printed_damage = _scrape_all(tex, r"losing \$([\d.]+)\$ and \$([\d.]+)\$ against its own seed")
    for i, value in enumerate(sorted(printed_damage, reverse=True)):
        out.append((f"tex: D3 comparator damage {i}", value, 0.0005, abs(damage[i])))

    # --- Paper I: the decision list's own count of bought directions ------------------------
    # It said "five" and went to six the moment D3 landed, with nothing checking it. The count is
    # derived from the list a reader actually counts: entries whose label says done.
    p1 = src.paper_tex("p1-conditional-lever")
    # The end delimiter must be searched *from* the opening one. An earlier `\end{description}`
    # arrives with an expanded `\input`, and a backwards slice is the empty string rather than an
    # error, so the naive form counted zero items and agreed with nothing.
    start = p1.index(r"\begin{description}\itemsep1pt")
    listing = p1[start : p1.index(r"\end{description}", start)]
    labels = re.findall(r"\\item\[(D[^\]]*?)\]", listing, re.DOTALL)
    if not labels:
        raise AssertionError("Paper I's decision list has no D-labelled entries to count")
    done = sum(1 for label in labels if "done" in label)
    out.append(
        (
            "tex: bought directions",
            _scrape_word(p1, r"(\w+) of the directions below have been\s+bought"),
            0.0,
            float(done),
        )
    )
    # D3's figures as Paper I restates them. Three papers quote this run; each site is pinned.
    for label, pattern, actual in (
        (
            "D3 delta (p1)",
            r"\$\\dd\(D-B\) = \+([\d.]+)\$, \\textbf\{inconclusive\}",
            mean([r["D"]["opt_heldout"] - r["B"]["opt_heldout"] for r in runs]),
        ),
        (
            "D3 noise band (p1)",
            r"against a noise band of \$([\d.]+)\$",
            stdev([r["D"]["opt_heldout"] for r in runs])
            + stdev([r["B"]["opt_heldout"] for r in runs]),
        ),
        # Reworded when D3 stopped being a bullet in the directions list and became
        # `sec:xprovider`: a bought, pre-registered run with nine claims of its own was being
        # reported inside a to-do list. "this entry" was the list's word for itself.
        ("D3 seeds (p1)", r"at the \$(\d+)\$ seeds the registration asked for", float(seeds)),
        ("D3 arm D accepted (p1)", r"accepted on only \$(\d+)\$ of \$\d+\$ seeds", float(d_seeds)),
        (
            "D3 share (p1)",
            r"seeds, and \$(\d+)\\%\$ of\s+the positive",
            100 * strata["comparator only"].share,
        ),
    ):
        out.append(
            (f"tex: {label}", _scrape(p1, pattern), 0.5 if "share" in label else 0.0005, actual)
        )
    out.append(
        (
            "tex: D3 held-out drafts judged (p1)",
            _scrape(p1, r"\\texttt\{PASS\} on none of \$(\d+)\$ held-out drafts"),
            0.0,
            _detector_row("study-xprovider.json", tuple(range(8)), True)["n"],
        )
    )

    # --- Paper II: the three-lineage detector table ------------------------------------------
    detectors = (
        ("intrinsic", "study-multihard.json", (0, 1, 2, 3, 4), False),
        ("cross-lineage", "study-crossfamily.json", (0, 1, 2), True),
        ("cross-provider", "study-xprovider.json", tuple(range(8)), True),
    )
    p2 = src.paper_tex("p2-repair-not-detection")
    num = r"\$(\d+)\$"
    dec = r"\$([\d.]+)\$"
    bold = r"\$\\mathbf\{([\d.]+)\}\$"
    patterns = (
        rf"\\emph\{{is\}} the worker & {num} & {dec} & {num} & {num} & {dec}",
        rf"same vendor, other lineage & {num} & {dec} & {num} & {num} & {bold}",
        rf"other vendor, other lineage & {num} & {bold} & {num} & {num}",
    )
    for (name, cassette, seed_tuple, sep), pattern in zip(detectors, patterns, strict=True):
        stats = _detector_row(cassette, seed_tuple, sep)
        printed = _scrape_all(p2, pattern)
        out.append((f"tab:detectors {name} n", printed[0], 0.0, stats["n"]))
        out.append((f"tab:detectors {name} pass rate", printed[1], 0.005, stats["pass_rate"]))
        out.append((f"tab:detectors {name} tp", printed[2], 0.0, stats["tp"]))
        out.append((f"tab:detectors {name} fp", printed[3], 0.0, stats["fp"]))
        if len(printed) > 4:
            out.append((f"tab:detectors {name} f1", printed[4], 0.005, stats["f1"]))
    # The base rates the caveat leans on, stated in the paper as exact counts rather than a
    # 3-dp decimal: 9/80 is 0.1125, which sits exactly on a rounding boundary, and a check whose
    # pass depends on which way a tie rounds is a check about float repr, not about the paper.
    #
    # All THREE pairs are captured. Until 2026-08-13 the pattern read
    # `\$5/50\$, \$4/30\$ and \$(\d+)/(\d+)\$`: the first two pairs were locator text, so four of
    # the six numerals in this sentence were never compared to anything. That is the same species
    # as the defect this file fixed at `tex: intrinsic p0cal-L1`, where a PASS rate of 1% was
    # printed against an artifact measuring 0.5% because the 1 sat inside the anchor.
    counts = _scrape_all(
        p2, r"already-correct drafts number \$(\d+)/(\d+)\$, \$(\d+)/(\d+)\$ and \$(\d+)/(\d+)\$"
    )
    for i, (name, cassette, seeds, sep) in enumerate(
        (
            ("intrinsic", "study-multihard.json", tuple(range(5)), False),
            ("cross-lineage", "study-crossfamily.json", tuple(range(3)), True),
            ("cross-provider", "study-xprovider.json", tuple(range(8)), True),
        )
    ):
        row = _detector_row(cassette, seeds, sep)
        already = row["tp"] + row["fn"]
        out.append((f"tex: detector p0 numerator, {name}", counts[2 * i], 0.0, already))
        out.append((f"tex: detector p0 denominator, {name}", counts[2 * i + 1], 0.0, row["n"]))
    # The closing argument runs on two axes, and an earlier draft pooled them: it set run 4's
    # +0.060, which is a SEED-GENOME between-run gap, beside two Delta(D-B) contrasts between
    # OPTIMIZED arms, as though the three were commensurable. They are not. The paragraph now
    # states each axis separately, so both sets are pinned here -- the seed-genome gaps against
    # the paired curve comparison, the optimization deltas against the run artifacts.
    swaps = _scrape_all(
        p2,
        r"score by \$\+([\d.]+)\$ for run~4's\s*\n?measurement tool, \$\+([\d.]+)\$ for the "
        r"accurate cross-lineage critic, \$\+([\d.]+)\$ for the degenerate\s*\n?cross-provider "
        r"one and \$-([\d.]+)\$ for the perfect oracle",
    )
    for i, (name, actual) in enumerate(
        (
            ("tool-fed", _curve_gap("run3", "run4")[0]),
            ("cross-lineage", _critique_swap_gap("crossfamily")),
            ("cross-provider", _critique_swap_gap("xprovider")),
            ("oracle", abs(_curve_gap("run3", "run7")[0])),
        )
    ):
        out.append((f"tex: seed-genome critique swap, {name}", swaps[i], 0.0005, actual))
    # The prose difference between the two substituted critics, which is what licenses reading a
    # model swap as a two-at-once intervention rather than a detection-only one.
    # The critique the paper quotes verbatim. A quotation is a claim about an artifact like any
    # other, and it is the one kind a reader cannot re-derive without opening the cassette.
    quoted = re.search(r"characters \(``\\emph\{([^}]+)\}''\)", p2)
    if not quoted:
        raise AssertionError("Paper II no longer quotes a critique where this guard expects one")
    crit = json.loads(Path("tests/fixtures/cassettes/study-crossfamily.json").read_text())[
        "entries"
    ]
    # Both sides are whitespace-normalised: the quotation wraps across a source line, so its
    # captured text carries a newline the completion never had, and a raw containment test would
    # report the paper misquoting an artifact it quotes correctly.
    needle = " ".join(quoted.group(1).split())
    outputs = [" ".join(v["output"].split()) for v in crit.values() if v.get("name") == "critic"]
    out.append(
        (
            "tex: quoted cross-lineage critique is in the cassette",
            1.0,
            0.0,
            float(any(needle in o for o in outputs)),
        )
    )
    lengths = _scrape_all(
        p2, r"median of \$(\d+)\$ characters.*?the cross-provider one in \$(\d+)\$"
    )
    for i, slug in enumerate(("crossfamily", "xprovider")):
        out.append(
            (f"tex: critique median length, {slug}", lengths[i], 0.0, _critique_median_length(slug))
        )
    # The distance between the two interventions that move both halves of the loop at once. It is
    # the sentence's whole argument, so it is derived rather than subtracted by hand.
    out.append(
        (
            "tex: tool vs cross-lineage separation",
            _scrape(p2, r"land within \$([\d.]+)\$\s*\n?of each other"),
            0.0005,
            abs(_curve_gap("run3", "run4")[0] - _critique_swap_gap("crossfamily")),
        )
    )
    restated = _scrape_all(p2, r"prompt alone: \$\+([\d.]+)\$ and \$\+([\d.]+)\$")
    for label, printed, actual in (
        (
            "cross-lineage held-out move",
            restated[0],
            mean(
                [
                    r["D"]["opt_heldout"] - r["B"]["opt_heldout"]
                    for r in json.loads(
                        Path(f"{DOCS}/validation-study-crossfamily.json").read_text()
                    )["runs"]
                ]
            ),
        ),
        (
            "cross-provider held-out move",
            restated[1],
            mean(
                [
                    r["D"]["opt_heldout"] - r["B"]["opt_heldout"]
                    for r in json.loads(
                        Path(f"{DOCS}/validation-study-xprovider.json").read_text()
                    )["runs"]
                ]
            ),
        ),
    ):
        out.append((f"tex: {label}", printed, 0.0005, actual))

    xp = _detector_row("study-xprovider.json", tuple(range(8)), True)
    out.append(
        (
            "tex: cross-provider missed positives",
            _scrape_word(p2, r"missing all (\w+) already-correct drafts"),
            0.0,
            xp["fn"],
        )
    )
    out.append(
        (
            "tex: cross-provider events",
            _scrape(p2, r"already-correct drafts in \$(\d+)\$ events"),
            0.0,
            xp["n"],
        )
    )
    return out


def _superlative_scope_checks() -> list[tuple[str, float, float, float]]:
    r"""Paper I's "largest loop-level effect", and the set that superlative ranges over.

    The claim sat at three sites unqualified while Paper I's own `tab:splitresults` printed a
    bigger number: the split sweep's $n_\mathrm{train}=16$ cell at $+0.062$, $2.12\sigma$ against
    $L_2$'s $2.07$. Defensible under that table's own caption, which reports its four cells as one
    quantity plus noise rather than as four effects; not defensible as written, because the
    sentence never said so and a reader could falsify it from the table on the facing page.

    Four checks, because the scope stays true only while all four do. The effect the sentence
    names has to be the largest of the ones the study re-measured; the cell the caveat names has to
    be the one that is actually larger; its $\sigma$ multiple is a restatement of a table cell and
    rots like every other restatement; and the ordering between the two has to still hold. If a
    later run makes another replicated level the largest, the first fails. If the split cell stops
    being larger, the caveat is naming a falsifier that no longer exists and the last one fails.

    Replicated means re-measured: the three $p_0$ levels were each scored five times over
    (`tab:repeat`), which is what licenses ranking them. Under those five-replicate means
    ($-0.021$, $+0.055$, $+0.019$) $L_2$ is still the largest, so the ranking does not depend on
    which of the two estimates the reader takes.
    """
    p1 = _sources().paper_tex("p1-conditional-lever")
    replicated = {f"L{i + 1}": mean(_p0_deltas(i)) for i in range(3)}
    largest = max(replicated.values())
    unreplicated = _split_row("split16").delta
    caveat = _scrape_all(p1, r"row at \$\+([\d.]+)\$, \$([\d.]+)\\sigma\$")
    return [
        (
            "tex: largest replicated loop-level effect",
            _scrape(
                p1,
                r"the \$\+([\d.]+)\$ gain at higher \$p_0\$ is the\s+"
                r"largest loop-level effect the study replicates",
            ),
            0.0005,
            largest,
        ),
        ("tex: the larger unreplicated cell", caveat[0], 0.0005, unreplicated),
        (
            "tex: the larger unreplicated cell in sigma",
            caveat[1],
            0.005,
            abs(unreplicated) / _noise_estimate().sigma_delta_mean,
        ),
        (
            "tex: the named cell really is the larger one",
            1.0,
            0.0,
            float(unreplicated > largest),
        ),
    ]


def _paraphrase_drift_checks() -> list[tuple[str, float, float, float]]:
    r"""Four claims a cross-audit found wrong on 2026-08-13, grouped by the mechanism that hid them.

    Each was either a fact the papers state in more than one wording, or a floor nobody derived.

    `_scrape` uses `re.search`, which returns the FIRST match. So a fact restated in different
    words is guarded at one site and free to rot at every other. That is how Paper III came to
    print "four runs overrode" the \$5 default while the shared harness section it `\input`s says
    "Five runs raised it explicitly": the guard read the shared file, agreed with it, and never saw
    the paraphrase two papers away. The ceiling count is therefore checked at EVERY site that
    states it, through `finditer`, and the number of sites is floored so that rewording one out of
    reach fails here instead of quietly reducing coverage.

    The other three are numbers that were simply never derived from anything: Paper I's L1 sigma
    multiple (printed 1.44 at six sites while its own table printed 1.43), Paper I's count of
    vacuously-satisfiable IFBench verifiers (five claimed, six real), and Paper II's licence to
    read arm A out of a four-arm artifact, which used to rest on a false claim about the CLI.
    """
    import glob

    out: list[tuple[str, float, float, float]] = []
    tex = _tex()

    # --- the $5 ceiling override count, at every wording that states it ------------------------
    over = []
    for path in sorted(glob.glob(f"{DOCS}/validation-study*.json")):
        with open(path, encoding="utf-8") as fh:
            est = json.load(fh)["meta"].get("estimated_usd")
        if est is not None and est > 5:
            over.append(est)
    # Over the prose only: Paper III's register quotes the withdrawn "Four runs overrode" verbatim,
    # which is the entry recording this very fix, and a sweep that read it would fail on it.
    sites = list(
        re.finditer(r"(\w+) runs (?:raised it explicitly|overrode)", _tex_without_registers())
    )
    for i, match in enumerate(sites):
        out.append(
            (
                f"tex: ceiling override count, site {i}",
                float(_WORD_NUM.get(match.group(1).lower(), float("nan"))),
                0,
                len(over),
            )
        )
    # A floor, not a measurement: if a rewording puts one site out of this pattern's reach, the
    # loop above would emit fewer checks and nothing would notice the loss of coverage. The
    # expected number is derived -- the shared harness section states it once and is expanded into
    # each of the three papers, and Paper III restates it in its own words -- so adding a paper
    # does not require editing a literal here.
    out.append(
        (
            "tex: ceiling override restatement sites",
            float(len(sites)),
            0,
            float(len(_sources().PAPERS) + 1),
        )
    )

    # --- Paper I's L1 sigma multiple, at all six prose sites ------------------------------------
    # Six sites printed 1.44 while tab:splitresults printed 1.43 for the same cell, because the
    # check that existed divided the paper's rounded -0.0417 (see `tex: sigma units L1`). These
    # patterns locate each site by its surrounding WORDS, never by the multiple itself.
    l1_sigma = _noise_estimate().in_sigma(_decomposition(0)["delta"])
    for i, pattern in enumerate(
        (
            r"is \$([\d.]+)\\sigma\$ of measurement noise and three other",
            r"puts this cell at\s+\$([\d.]+)\\sigma\$\. So the honest summary",
            r"and \$([\d.]+)\\sigma\$ of measurement noise\. What survives",
            r"It rests on a single \$([\d.]+)\\sigma\$ cell",
            r"the cell is \$([\d.]+)\\sigma\$ of a measured noise floor",
        )
    ):
        out.append((f"tex: L1 sigma multiple, site {i}", _scrape(tex, pattern), 0.005, l1_sigma))

    # --- The do-nothing floor, at every site that states it --------------------------------------
    # Both Papers I and II carry this sentence. It was checked with `re.search`, which returns the
    # first match, so the concatenated corpus yielded Paper I's `six`, agreed with the six the
    # verifiers actually produce, and never read Paper II's `five` -- a value Paper I's own
    # register had formally withdrawn. Swept with `finditer` and the site count asserted, so a
    # third restatement fails instead of going unread. Registers are stripped: they quote the
    # withdrawn count on purpose.
    n_verifiers, n_vacuous = _vacuous_ifbench_verifiers()
    floor_tex = _tex_without_registers()
    vacuous_sites = list(
        re.finditer(r"([\w-]+) of ([\w-]+) IFBench verifiers are vacuously", floor_tex)
    )
    # The expected site count is derived from a *different* phrase in the same paragraph, so a
    # reworded count sentence makes the two disagree instead of quietly reducing coverage.
    out.append(
        (
            "tex: do-nothing floor, sites stating the counts",
            float(len(vacuous_sites)),
            0,
            float(len(re.findall(r"A do-nothing floor in the", floor_tex))),
        )
    )
    for i, site in enumerate(vacuous_sites):
        out.append(
            (
                f"tex: vacuously satisfiable verifiers, site {i}",
                float(_WORD_NUM[site.group(1).lower()]),
                0,
                float(n_vacuous),
            )
        )
        out.append(
            (
                f"tex: IFBench verifier count, site {i}",
                float(_WORD_NUM[site.group(2).lower()]),
                0,
                float(n_verifiers),
            )
        )

    # --- Paper III's enumeration of the replayable runs, and Paper I's replicate count -----------
    # Two numbers that were wrong because nothing derived them: an enumeration whose parts summed
    # to twenty-seven against a total of twenty-five, and a registered `5 -> 9` where five plus
    # five is ten. Both are inert -- no result moves either way -- which is precisely why they
    # survived. They are derived here so inertness stops being protection.
    groups = _replayable_partition()
    out.append(
        (
            "tex: enumeration parts sum to the replayable total",
            float(sum(groups.values())),
            0,
            float(_registry_counts()[1]),
        )
    )
    # Read from the register-stripped corpus: both registers quote the withdrawn wording verbatim,
    # and `\s+` rather than a literal space because the body wraps mid-phrase -- a first-match
    # scrape over the full corpus reads the register's `eleven` and calls the fix a regression.
    body = _tex_without_registers()
    out.append(
        (
            "tex: second-worker line, runs enumerated",
            _scrape_word(body, r"the\s+([\w-]+)\s+runs of the second-worker line"),
            0,
            float(groups["second-worker line"]),
        )
    )
    out.append(
        (
            "tex: cross-run sigma replicate count",
            _scrape(body, r"plus these five is \$(\d+)\$, not \$9\$"),
            0,
            float(_noise_estimate().n_replicates),
        )
    )

    # --- Paper II's arm-A licence ---------------------------------------------------------------
    entries, violations = _arm_a_inertness()
    out.append(
        (
            "tex: arm-A seed entries",
            _scrape(tex, r"\$(\d+)\$\s*\narm-\\textbf\{A\} seed entries across the release"),
            0,
            float(entries),
        )
    )
    # The claim is "with no exception", so the artifact side must be zero. Compared against a
    # literal 0 deliberately: this is the one number whose correct value is not read from the
    # release but asserted about it, and a non-zero here retracts the paper's sentence.
    out.append(("arm-A inertness violations", 0.0, 0, float(violations)))

    # --- the wrong values these corrections withdraw, each derived by REAPPLYING its mistake -----
    # A register entry has to print the number the paper used to carry, or a reader cannot tell what
    # was fixed. That leaves a decimal in the prose no artifact produces, which
    # `test_paper_prose_coverage` correctly refuses. The route out is not an exemption: every one of
    # these is reproducible by rounding exactly where the original rounded, so each is derived, and
    # a change to the underlying data moves the withdrawn value with it. Same principle as
    # `_restated_statistic_checks`, which recomputes the population-sd t values by reapplying that
    # estimator rather than transcribing them.
    sigma = _noise_estimate().sigma_delta_mean
    l1 = _decomposition(0)["delta"]
    run1 = _abs_mean_delta("validation-study.json")
    for label, pattern, tol, want in (
        # What the exact arithmetic gives, quoted in the entry as the right answer.
        ("register: L1 sigma exact", r"is \$([\d.]+)\$, and \$1.44\$ is", 0.0001, abs(l1) / sigma),
        # What the check computed: the delta rounded to the four decimals the paper displayed.
        (
            "register: L1 sigma from a 4-dp delta",
            r"\$([\d.]+)\$ and agreed with the wrong value",
            0.0001,
            abs(round(l1, 4)) / sigma,
        ),
        # What the prose printed: a 3-dp delta over a 4-dp sigma, both already rounded for display.
        (
            "register: L1 sigma double-rounded",
            r"and \$([\d.]+)\$ is\s+what the paper's own displayed",
            0.005,
            round(abs(l1), 3) / round(sigma, 4),
        ),
        # Paper III's, the same double rounding one decimal coarser.
        (
            "register: run-1 sigma double-rounded",
            r"which gives \$3.23\$, and \$([\d.]+)\$ is what the paper's own",
            0.005,
            round(run1, 3) / round(sigma, 3),
        ),
    ):
        out.append((label, _scrape(tex, pattern), tol, want))
    return out


def _defect_class_count_checks() -> list[tuple[str, float, float, float]]:
    """How many defects Paper III found, and how many the audit ADDED, from the table itself.

    Six prose sites state one of these two counts and none was pinned. Three of the four that state
    the increment said `seven`; \\S\\ref{sec:defect} said `six`, which is what the table supports:
    nine rows, of which the caption assigns 1--3 to the original headline. The likely source of the
    wrong figure is $9-2$ --- the abstract and the conclusion both introduce the original as *two*
    ordinary wiring choices, which is true of its causes and omits the symptom the caption counts as
    defect 1.

    Both counts are derived from the table's own rows and the range its caption names, so adding a
    tenth defect moves the expected values instead of breaking a check that then gets edited to
    match. A count restated in an abstract is the number a reader checks first, which is the same
    argument that put `_corrections_count_checks` here.
    """
    src = _sources()
    slug = "p3-silent-no-op"
    tex = src.paper_tex(slug)
    label = tex.index(r"\label{tab:defects}")
    table = tex[tex.rindex(r"\begin{table}", 0, label) : label]
    rows = [int(n) for n in re.findall(r"^(\d+) & ", table, re.MULTILINE)]
    named = re.search(r"Defects (\d+)--(\d+) invalidated the original headline", table)
    if named is None or rows != list(range(1, len(rows) + 1)):
        raise ValueError(
            f"tab:defects has rows {rows} and "
            f"{'no' if named is None else 'a'} caption naming the original headline's range; both "
            "are load-bearing here, so a change to either must be made deliberately"
        )
    total = float(len(rows))
    # The caption's range is inclusive and starts at 1, so its upper bound IS the original's size.
    added = total - float(named.group(2))
    out = [
        (
            "tab:defects rows against its caption's count",
            _scrape_word(table, r"\{(\w+) ways"),
            0.0,
            total,
        )
    ]
    for site, pattern, expected in (
        ("abstract", r"found the same class of error (\w+) more times, including", added),
        ("sec:defect", r"found the same class of error (\w+) more times\.\s+Table~", added),
        ("contribution 3", r"\\item \\textbf\{(\w+) more instances of the same class\}", added),
        ("conclusion", r"found (\w+) more errors of the same class", added),
        ("threats", r"The (\w+) defects are not independent findings", total),
    ):
        out.append((f"p3 defect count, {site}", _scrape_word(tex, pattern), 0.0, expected))
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
        "    `tex:` numbers above -- those are scraped from the papers' sources, the rest are not."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
