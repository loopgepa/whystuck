#!/usr/bin/env python
"""Did the OPTIMIZER's evolved critique gain discrimination, or only move its threshold?

Zero cost, offline. This is the sharpest question the corrected-harness re-run makes
answerable, and it closes a gap the C1 ablation could only half-address.

C1 showed that a verify gene written *by hand* against this study's own findings changed
only *when* the detector fires (PASS rate 0.09 -> 0.00) and never *whether it can tell* --- it
produced no true positive at either setting. That was one author's single point in prose
space, and the obvious objection is that a search over prose might find what one person could
not. The re-run answers it directly: arm D's evolved genes are real, substantial rubrics
(~3000 characters against a 127-character seed), so we can score them as detectors on exactly
the drafts they judged and compare against the seed rubric on the same drafts.

If the evolved gene also only moves the threshold, then C1's mechanism holds for the search
and not merely for the author, and the loop-level null has an explanation that does not
involve the optimizer failing at its job: there is no discrimination in this prose space to
find.

    python scripts/analyze_evolved_gene.py [RESULTS.json CASSETTE.json]...
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_critique_detector import collect_events, detector_stats  # noqa: E402

RESULTS = "docs/validation-study-c2b.json"
CASSETTE = "tests/fixtures/cassettes/study-c2b.json"
VERIFY_GENE = "ensure_correct"
LOOP_KW: dict = {}


@dataclass(frozen=True)
class Condition:
    """One critique gene scored as a detector on the drafts it actually judged."""

    seed: int
    kind: str  # "seed" (the seed rubric) or "evolved" (arm D's)
    stats: dict[str, float]

    @property
    def n(self) -> int:
        return int(self.stats["n"])

    @property
    def pass_rate(self) -> float:
        return (self.stats["tp"] + self.stats["fp"]) / self.stats["n"] if self.stats["n"] else 0.0


def score_run(results: str, cassette: str, *, n_heldout: int = 10) -> list[Condition]:
    """Score the seed rubric and arm D's evolved rubric, per seed, for one released run.

    Parameterised rather than hardcoded because the split-size sweep's load-bearing
    pre-registered prediction is exactly this quantity on three further runs, and a scorer
    that only reads one run would have to be copied to check it.

    A critique can only be replayed against the drafts it actually judged, so each condition
    overrides the WHOLE genome that produced it: the seed rubric judging seed drafts, and the
    evolved rubric judging the evolved generator's drafts. That is not a same-drafts contrast
    -- no such recording exists, because the two genes moved together -- but it is the
    comparison that answers whether the evolved detector discriminates at all.
    """
    runs = json.loads(Path(results).read_text())["runs"]
    # Arm C never evolves the verify gene, so it carries the seed text; the sweep runs arms
    # BD only, in which case the seed text has to come from the study's seed genome instead.
    holder = next((r for r in runs if isinstance(r.get("C"), dict)), None)
    if holder is not None:
        seed_text = holder["C"]["final_genes"][VERIFY_GENE]
    else:
        from whystuck.eval.study import seed_genome

        genome = seed_genome("study-A", evolvable=set())
        seed_text = next(g.content for g in genome.genes if g.id == VERIFY_GENE)

    out: list[Condition] = []
    for seed, run in enumerate(runs):
        conditions: tuple[tuple[str, dict[str, str]], ...] = (
            ("seed", {VERIFY_GENE: str(seed_text)}),
            ("evolved", {k: str(v) for k, v in run["D"]["final_genes"].items()}),
        )
        for kind, overrides in conditions:
            events = collect_events(
                cassette, [seed], loop_kw=LOOP_KW, gene_overrides=overrides, n_heldout=n_heldout
            )
            if not events:
                raise ValueError(
                    f"{results} seed {seed} ({kind}): no recorded events in {cassette}. A "
                    "PASS rate computed from nothing is 0.0, which is exactly the number the "
                    "finding predicts, so an empty replay must raise rather than report"
                )
            out.append(Condition(seed=seed, kind=kind, stats=detector_stats(events)))
    return out


def _report(results: str, cassette: str) -> None:  # pragma: no cover - CLI convenience
    rows = score_run(results, cassette)
    print(f"=== {results} : the critique as a failure detector, per seed ===")
    print(
        f"{'gene':>22} {'n':>4} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'PASS rate':>10} {'F1':>6}"
    )
    for r in rows:
        s, f1 = r.stats, r.stats["f1"]
        label = f"seed{r.seed} {'arm A (seed)' if r.kind == 'seed' else 'arm D (evolved)'}"
        print(
            f"{label:>22} {r.n:>4} {int(s['tp']):>4} {int(s['fp']):>4} "
            f"{int(s['fn']):>4} {int(s['tn']):>4} {r.pass_rate:>10.2f} "
            f"{'undef' if f1 != f1 else format(f1, '.2f'):>6}"
        )
    ev = [r for r in rows if r.kind == "evolved"]
    sd = [r for r in rows if r.kind == "seed"]
    tp_ev = int(sum(r.stats["tp"] for r in ev))
    tp_sd = int(sum(r.stats["tp"] for r in sd))
    never = sum(1 for r in ev if r.pass_rate == 0.0)
    print(
        f"\nTrue positives, pooled: seed rubric {tp_sd}, evolved {tp_ev}."
        f"\nEvolved genes with a PASS rate of exactly 0: {never}/{len(ev)}."
        "\nA detector that never emits a true positive has no discrimination to offer,"
        "\nwhatever its threshold; if both are zero, the optimizer searched a space in"
        "\nwhich the reachable moves are threshold moves, exactly as C1 found by hand.\n"
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    args = list(argv if argv is not None else sys.argv[1:])
    pairs = (
        [(args[i], args[i + 1]) for i in range(0, len(args), 2)] if args else [(RESULTS, CASSETTE)]
    )
    if any(len(p) != 2 or not p[1] for p in pairs):
        print("usage: analyze_evolved_gene.py [RESULTS.json CASSETTE.json]...", file=sys.stderr)
        return 2
    for results, cassette in pairs:
        _report(results, cassette)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
