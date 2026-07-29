#!/usr/bin/env python
"""Split Δ(D−B) into the treated component and the shared one (zero cost, offline).

Δ(D−B) is the paper's headline, and it is a *whole-system* contrast. But the arms do not
differ in one component: arm~B evolves `generate`, and arm~D evolves `generate` **and**
`ensure_correct`. Only the second gene is the treatment. So any gap between the arms mixes
two things — a different draft from a differently-evolved shared gene, and a different loop
built on top of that draft — and the aggregate cannot say which one moved.

The split is an identity, not a model. Writing d for the mean first-draft score and f for
the mean final score of an arm,

    Δ = f_D − f_B = (d_D − d_B) + [(f_D − d_D) − (f_B − d_B)]
                     \\_________/   \\___________________________/
                      draft term            loop term

The draft term is attributable to `generate`, which both arms may evolve; the loop term is
what the critique-and-refine machinery adds on top of the draft it was handed, which is
where the treatment lives. Reporting Δ alone credits the treatment with the draft term.

The second output is a stratification. p₀ is the base rate of already-correct first drafts,
and the mechanism the paper proposes for a negative Δ is that a never-accepting critique
sends already-correct drafts to a refine round that can break them — a cost carried
entirely by the already-correct stratum, whose weight is p₀. That is checkable: partition
held-out tasks by whether the **seed genome's** first draft was correct, and read the two
strata separately. The stratifier must be the seed genome's draft and not the arm's own,
because an arm's draft is downstream of the treatment and conditioning on it would
condition on a collider; the seed draft is a pre-treatment covariate and is identical for
both arms by construction.

Both outputs replay committed cassettes and combine them with the released per-task
held-out vectors, so nothing here costs money and nothing here is re-derived from a
different metric than the runs reported.

    python scripts/analyze_delta_decomposition.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from whystuck.eval.cassette import Cassette, CassetteMiss
from whystuck.eval.llm_worker import make_self_refine_run_task
from whystuck.eval.multi_constraint import load_multi_constraint, mc_feedback, mc_pass, mc_score
from whystuck.eval.study import seed_genome

CASSETTE_DIR = "tests/fixtures/cassettes"
DOCS = "docs"
_N_HELDOUT = 10


@dataclass(frozen=True)
class Level:
    """One p₀ level: a released run plus the exclusions that generated its task pool."""

    name: str
    slug: str
    exclude: frozenset[str]

    @property
    def results(self) -> str:
        return f"{DOCS}/validation-study-{self.slug}.json"

    @property
    def cassette(self) -> str:
        return f"{CASSETTE_DIR}/study-{self.slug}.json"


#: The three levels of the p₀ sweep. L1 is the four-arm re-run itself.
LEVELS = (
    Level("L1", "c2b", frozenset()),
    Level("L2", "p0-L2", frozenset({"exact_word_count"})),
    Level("L3", "p0-L3", frozenset({"exact_word_count", "word_max_length"})),
)

#: The rows of the split-size sweep. n_train=8 is the L1 row above, so it is not repeated.
#: Same decomposition, a different axis: these vary the TRAIN split with the held-out set held
#: fixed, so the draft/loop split answers whether a bigger split rescues the treated component
#: or only the shared gene.
SPLIT_LEVELS = (
    Level("n=4", "split4", frozenset()),
    Level("n=16", "split16", frozenset()),
    Level("n=24", "split24", frozenset()),
)

#: First `n_train` whose pool is no longer 64. `arm_rows` derives the generated pool from
#: `n_heldout` alone, which is correct only while `2*n_train + 2 <= 64`; above it the run's own
#: pool is larger, the shuffle is seeded differently, and a replay would silently score a
#: DIFFERENT held-out set while reporting a confident number. That is this paper's subject
#: matter, so it is a guard and not a comment.
_POOL_CLIFF_N_TRAIN = 32


@dataclass(frozen=True)
class Row:
    """One held-out task under one arm's optimized genome."""

    seed: int
    task_id: str
    draft: float
    final: float
    accepted: bool
    seed_draft_correct: bool
    draft_words: int
    final_words: int


@dataclass(frozen=True)
class Decomposition:
    level: str
    n: int
    draft_b: float
    draft_d: float
    final_b: float
    final_d: float

    @property
    def delta(self) -> float:
        return self.final_d - self.final_b

    @property
    def draft_term(self) -> float:
        """The part of Δ attributable to `generate`, which BOTH arms evolve."""
        return self.draft_d - self.draft_b

    @property
    def loop_term(self) -> float:
        """The part of Δ attributable to the loop built on that draft — the treatment."""
        return (self.final_d - self.draft_d) - (self.final_b - self.draft_b)


@dataclass(frozen=True)
class Stratum:
    n: int
    n_b: int
    n_d: int
    mean_b: float
    mean_d: float
    weight: float

    @property
    def gap(self) -> float:
        return self.mean_d - self.mean_b

    @property
    def contribution(self) -> float:
        return self.gap * self.weight


def _unreachable(_prompt: str) -> str:
    # Never invoked: Cassette.wrap raises CassetteMiss before calling through.
    raise AssertionError("unreachable in replay mode")


#: `arm_rows`'s pseudo-arm for the unoptimized seed genome. It is not a study arm: it reads
#: arm~B's *seed* per-task vector, which every arm shares because the seed genome is shared,
#: and it is what isolates the refine step's own sign with no optimizer in the picture.
SEED_ARM = "SEED"


def arm_rows(level: Level, arm: str) -> list[Row]:
    """Replay one arm's genome over held-out and pair drafts with final scores.

    The final score is the run's own released per-task value rather than a re-scored
    trace, so the decomposition is anchored to the number the paid run reported. The draft
    score is recomputed from the recorded first completion, which is what the cassette is
    for. Pass `SEED_ARM` for the unoptimized seed genome.
    """
    doc = json.loads(Path(level.results).read_text())
    runs = doc["runs"]
    n_train = doc["meta"].get("n_train")
    if isinstance(n_train, int) and n_train >= _POOL_CLIFF_N_TRAIN:
        raise ValueError(
            f"{level.results} used n_train={n_train}, at or past the pool cliff of "
            f"{_POOL_CLIFF_N_TRAIN}: this replay derives the generated pool from n_heldout "
            "alone, so above the cliff it would score a DIFFERENT held-out set than the run "
            "did and report a confident decomposition of the wrong tasks"
        )
    if doc["meta"].get("n_heldout") not in (None, _N_HELDOUT):
        raise ValueError(
            f"{level.results} used n_heldout={doc['meta']['n_heldout']}, not {_N_HELDOUT}; the "
            "replay would load a different held-out split than the run scored"
        )
    pool = max(64, 2 * _N_HELDOUT + 2)
    rows: list[Row] = []
    with Cassette(level.cassette, mode="replay") as cas:
        for seed, run in enumerate(runs):
            heldout = load_multi_constraint(
                "heldout", seed=seed, n=_N_HELDOUT, pool=pool, hard=True, exclude=level.exclude
            )
            if arm == SEED_ARM:
                final_by_task = run["B"]["seed_heldout_by_task"]
                overrides: dict[str, str] = {}
            else:
                final_by_task = run[arm]["opt_heldout_by_task"]
                overrides = dict(run[arm]["final_genes"])
            genomes = {
                "arm": seed_genome("study-A", evolvable=set(), overrides=overrides),
                # The pre-treatment stratifier: the seed genome, identical across arms.
                "seed": seed_genome("study-A", evolvable=set()),
            }
            run_task = make_self_refine_run_task(
                cas.wrap(_unreachable, name="worker"),
                mc_score,
                feedback=mc_feedback,
                max_retries=1,
            )
            for task in heldout:
                if task.id not in final_by_task:
                    continue
                try:
                    arm_trace = run_task(genomes["arm"], task)
                    seed_trace = run_task(genomes["seed"], task)
                except CassetteMiss:
                    # A genuinely un-recorded call is skippable; a code regression is not,
                    # and would surface as a wrong n rather than a silent pass.
                    continue
                gen0 = next((s for s in arm_trace.steps if s.step_id == "generate-0"), None)
                crit0 = next((s for s in arm_trace.steps if s.step_id == "critique-0"), None)
                seed_gen0 = next((s for s in seed_trace.steps if s.step_id == "generate-0"), None)
                if gen0 is None or crit0 is None or seed_gen0 is None:
                    continue
                # The answer the loop ends on is the last step that produced one, which is
                # generate-0 when the critique accepted and the final refine otherwise.
                answered = [
                    s for s in arm_trace.steps if s.step_id.startswith(("generate", "refine"))
                ]
                rows.append(
                    Row(
                        seed=seed,
                        task_id=task.id,
                        draft=mc_score(gen0.output_summary or "", task),
                        final=final_by_task[task.id],
                        accepted=(crit0.output_summary or "").strip().upper().startswith("PASS"),
                        seed_draft_correct=mc_pass(seed_gen0.output_summary or "", task),
                        draft_words=len((gen0.output_summary or "").split()),
                        final_words=len((answered[-1].output_summary or "").split()),
                    )
                )
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError(
            "mean of an empty replay: a level with no recorded held-out events would "
            "report 0.0 and read as 'no effect' rather than 'nothing was measured'"
        )
    return sum(values) / len(values)


def decompose(level: Level) -> Decomposition:
    """Δ = draft term + loop term for one p₀ level."""
    b, d = arm_rows(level, "B"), arm_rows(level, "D")
    if len(b) != len(d):
        raise ValueError(
            f"{level.name}: arms replayed a different number of held-out tasks "
            f"(B={len(b)}, D={len(d)}); a decomposition across unequal task sets would "
            "compare different benchmarks"
        )
    return Decomposition(
        level=level.name,
        n=len(b),
        draft_b=_mean([r.draft for r in b]),
        draft_d=_mean([r.draft for r in d]),
        final_b=_mean([r.final for r in b]),
        final_d=_mean([r.final for r in d]),
    )


def strata(level: Level) -> dict[bool, Stratum]:
    """Split Δ by whether the SEED genome's first draft was already correct.

    Keyed by that boolean. The already-correct stratum has weight p₀ by construction, and
    is the stratum the always-FIX mechanism says the loss lives on.
    """
    b, d = arm_rows(level, "B"), arm_rows(level, "D")
    by_key_b = {(r.seed, r.task_id): r for r in b}
    by_key_d = {(r.seed, r.task_id): r for r in d}
    shared = sorted(set(by_key_b) & set(by_key_d))
    total = len(shared)
    out: dict[bool, Stratum] = {}
    for correct in (True, False):
        keys = [k for k in shared if by_key_b[k].seed_draft_correct is correct]
        if not keys:
            out[correct] = Stratum(0, 0, 0, float("nan"), float("nan"), 0.0)
            continue
        out[correct] = Stratum(
            n=len(keys),
            n_b=len(keys),
            n_d=len(keys),
            mean_b=_mean([by_key_b[k].final for k in keys]),
            mean_d=_mean([by_key_d[k].final for k in keys]),
            weight=len(keys) / total,
        )
    return out


def released_delta(level: Level) -> float:
    """Δ(D−B) as the paid run itself reported it, from the arm-level means."""
    runs = json.loads(Path(level.results).read_text())["runs"]
    d = _mean([r["D"]["opt_heldout"] for r in runs])
    b = _mean([r["B"]["opt_heldout"] for r in runs])
    return d - b


def answer_length(level: Level, arm: str) -> tuple[float, float]:
    """(mean draft words, mean final-answer words) for one arm.

    This closes the do-nothing-floor threat rather than restating it. The metric is a
    fraction of constraints satisfied and is not anchored at zero for an empty answer, so a
    critique that pushed toward saying less would score better without answering better. If
    that were happening, the arm whose critique never accepts would end on systematically
    shorter answers than prompt-only.
    """
    rows = arm_rows(level, arm)
    return (
        _mean([float(r.draft_words) for r in rows]),
        _mean([float(r.final_words) for r in rows]),
    )


def refine_productivity(level: Level, arm: str) -> tuple[int, int, float]:
    """(n_accepted, n_refined, mean gain per refined task) for one arm.

    Read with care: an arm that accepts some drafts refines a *selected* subset of the
    rest, so a per-refine gain is not comparable across arms without that caveat. Passing
    `SEED_ARM` gives the refine step's own sign, with no optimizer involved.
    """
    rows = arm_rows(level, arm)
    refined = [r for r in rows if not r.accepted]
    accepted = [r for r in rows if r.accepted]
    gain = _mean([r.final - r.draft for r in refined]) if refined else float("nan")
    return len(accepted), len(refined), gain


def _available(levels: tuple[Level, ...]) -> tuple[Level, ...]:
    """Levels whose released run is present, announcing any that are not.

    The split-size sweep rows are registered here before they are bought, so a missing file
    means "not run yet" rather than "broken". Saying so beats both crashing and silently
    printing a shorter table that reads as complete.
    """
    have = tuple(lv for lv in levels if Path(lv.results).exists())
    for lv in levels:
        if lv not in have:
            print(f"  (skipping {lv.name}: {lv.results} not present -- run not bought yet)")
    return have


def main() -> int:  # pragma: no cover - CLI convenience
    print("=== Δ(D−B) split into the shared gene and the treated one ===")
    print("draft term = `generate` (both arms evolve it); loop term = the treatment.\n")
    groups = (("p0 sweep", _available(LEVELS)), ("split-size sweep", _available(SPLIT_LEVELS)))

    print(f"{'axis':16} {'level':6} {'n':>3} {'Δ(D−B)':>9} {'draft':>9} {'loop':>9}  treated")
    for axis, levels in groups:
        for level in levels:
            d = decompose(level)
            # A share is only an apportionment when the two terms ADD. If they oppose, the
            # loop term can exceed Δ itself and the "share" prints as 157%, which reads as a
            # precision claim rather than as cancellation. Report the sign story instead.
            same_sign = (d.draft_term >= 0) == (d.loop_term >= 0)
            if abs(d.delta) < 0.01:
                share = f"{'n/a':>8}"
            elif not same_sign:
                share = f"{'oppose':>8}"
            else:
                share = f"{abs(d.loop_term) / abs(d.delta):>8.0%}"
            print(
                f"{axis:16} {d.level:6} {d.n:>3} {d.delta:>+9.4f} {d.draft_term:>+9.4f} "
                f"{d.loop_term:>+9.4f} {share}"
            )

    print("\n=== the same Δ, stratified on the SEED genome's first draft (pre-treatment) ===")
    print("The always-FIX mechanism predicts the loss sits in the ALREADY-CORRECT stratum.\n")
    print(
        f"{'axis':16} {'level':6} {'stratum':16} {'n':>3} {'B':>7} {'D':>7} "
        f"{'D−B':>8} {'contrib':>9}"
    )
    for axis, levels in groups:
        for level in levels:
            for correct in (True, False):
                g = strata(level)[correct]
                if not g.n:
                    continue
                label = "already correct" if correct else "already wrong"
                print(
                    f"{axis:16} {level.name:6} {label:16} {g.n:>3} {g.mean_b:>7.4f} "
                    f"{g.mean_d:>7.4f} {g.gap:>+8.4f} {g.contribution:>+9.4f}"
                )

    print("\n=== what the refine rounds bought ===")
    print("SEED is the unoptimized genome: it gives the refine step's own sign. Selection")
    print("caveat on B vs D: an arm that accepts more refines a different, selected subset.\n")
    print(f"{'axis':16} {'level':6} {'arm':6} {'accepted':>9} {'refined':>8} {'gain/refined':>13}")
    for axis, levels in groups:
        for level in levels:
            for arm in (SEED_ARM, "B", "D"):
                acc, ref, gain = refine_productivity(level, arm)
                print(f"{axis:16} {level.name:6} {arm:6} {acc:>9} {ref:>8} {gain:>+13.3f}")

    print("\n=== answer length by arm: is the do-nothing floor being exploited? ===")
    print("A critique pushing toward shorter answers would put D well below B.\n")
    print(f"{'axis':16} {'level':6} {'arm':6} {'draft words':>12} {'final words':>12} {'Δ':>8}")
    for axis, levels in groups:
        for level in levels:
            for arm in ("B", "D"):
                d0, d1 = answer_length(level, arm)
                print(f"{axis:16} {level.name:6} {arm:6} {d0:>12.1f} {d1:>12.1f} {d1 - d0:>+8.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
