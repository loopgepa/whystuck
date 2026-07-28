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

    python scripts/analyze_evolved_gene.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_critique_detector import collect_events, detector_stats  # noqa: E402

RESULTS = "docs/validation-study-c2b.json"
CASSETTE = "tests/fixtures/cassettes/study-c2b.json"
VERIFY_GENE = "ensure_correct"
LOOP_KW: dict = {}


def main() -> int:  # pragma: no cover - CLI convenience
    runs = json.loads(Path(RESULTS).read_text())["runs"]
    seed_text = runs[0]["C"]["final_genes"][VERIFY_GENE]  # arm C never changes it

    print("=== the evolved critique as a failure detector (arm D, per seed) ===")
    print(f"seed rubric: {len(seed_text)} chars\n")
    print(
        f"{'gene':>22} {'n':>4} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4} {'PASS rate':>10} {'F1':>6}"
    )

    # A critique can only be replayed against the drafts it actually judged, so each
    # condition overrides the WHOLE genome that produced it. Arm A is the seed rubric judging
    # seed drafts; arm D is the evolved rubric judging the evolved generator's drafts. That
    # is not a same-drafts contrast -- no such recording exists, because the two genes moved
    # together -- but it is the comparison that answers whether the evolved detector
    # discriminates at all.
    rows = []
    for i, run in enumerate(runs):
        conditions = (
            (f"seed{i} arm A (seed)", {VERIFY_GENE: seed_text}),
            (f"seed{i} arm D (evolved)", dict(run["D"]["final_genes"])),
        )
        for label, overrides in conditions:
            events = collect_events(CASSETTE, [i], loop_kw=LOOP_KW, gene_overrides=overrides)
            if not events:
                print(f"{label:>22}    (no recorded events)")
                continue
            s = detector_stats(events)
            pass_rate = (s["tp"] + s["fp"]) / s["n"] if s["n"] else float("nan")
            f1 = s["f1"]
            print(
                f"{label:>22} {int(s['n']):>4} {int(s['tp']):>4} {int(s['fp']):>4} "
                f"{int(s['fn']):>4} {int(s['tn']):>4} {pass_rate:>10.2f} "
                f"{'undef' if f1 != f1 else format(f1, '.2f'):>6}"
            )
            rows.append((label, s, pass_rate))

    ev = [r for r in rows if "evolved" in r[0]]
    sd = [r for r in rows if "arm A" in r[0]]
    if ev and sd:
        tp_ev = sum(r[1]["tp"] for r in ev)
        tp_sd = sum(r[1]["tp"] for r in sd)
        print(
            f"\nTrue positives, pooled: seed rubric {int(tp_sd)}, evolved {int(tp_ev)}."
            "\nA detector that never emits a true positive has no discrimination to offer,"
            "\nwhatever its threshold; if both are zero, the optimizer searched a space in"
            "\nwhich the reachable moves are threshold moves, exactly as C1 found by hand."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
