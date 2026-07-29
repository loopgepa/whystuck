#!/usr/bin/env python
"""Measure p₀ --- the base rate of already-correct first drafts --- per exclusion level.

Zero cost, offline. p₀ is the axis the pre-registered sweep varies, and it is the quantity the
always-\\texttt{FIX} mechanism is a function of: a critique that never accepts can only lose
where a draft needed no repair, so p₀ bounds what that policy costs on the training signal. At
the base level only 2 of 30 first drafts were already correct, which is why the degenerate
candidate is nearly free there.

p₀ needs no dedicated purchase. Every recorded cassette already contains it, because the first
critique event judged the first draft: replay the loop, take the draft the first critique saw,
and score it. What each level costs is one cheap arm-A pass to *create* the cassette
(~\\$0.09), not a measurement run.

The independence estimate in the paper (multiply per-constraint satisfaction rates over each
task) is biased HIGH -- it predicts 0.127 at the base level against a measured 0.067 -- because
constraint failures correlate: a draft that miscounts its words tends to break other
constraints too. So the estimate orders the levels and this script sets them.

    python scripts/measure_p0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_critique_detector import base_rate_p0, collect_events, detector_stats  # noqa: E402

CASSETTES = "tests/fixtures/cassettes"
SEEDS = [0, 1, 2]
N_HELDOUT = 10

#: (label, cassette, excluded constraints). The base level is the four-arm re-run's own
#: cassette, so it costs nothing; the others come from the calibration passes.
LEVELS: list[tuple[str, str, frozenset[str]]] = [
    ("L1  none (four-arm re-run)", f"{CASSETTES}/study-c2b.json", frozenset()),
    ("L2  -exact_word_count", f"{CASSETTES}/study-p0cal-L2.json", frozenset({"exact_word_count"})),
    (
        "L3  -exact_word_count -word_max_length",
        f"{CASSETTES}/study-p0cal-L3.json",
        frozenset({"exact_word_count", "word_max_length"}),
    ),
]


def main() -> int:  # pragma: no cover - CLI convenience
    print(f"{'level':40} {'n':>4} {'p0':>7}  {'PASS rate':>9}  seed-rubric detector")
    rows = []
    for label, cassette, exclude in LEVELS:
        if not Path(cassette).exists():
            print(f"{label:40} {'--':>4} {'--':>7}  (no cassette yet: {cassette})")
            continue
        events = collect_events(cassette, SEEDS, loop_kw={}, n_heldout=N_HELDOUT, exclude=exclude)
        p0 = base_rate_p0(events)
        stats = detector_stats(events)
        pass_rate = sum(1 for verdict, _ in events if verdict) / len(events)
        rows.append((label, p0))
        print(
            f"{label:40} {len(events):>4} {p0:>7.3f}  {pass_rate:>9.2f}  "
            f"tp={stats['tp']:.0f} fp={stats['fp']:.0f} "
            f"fn={stats['fn']:.0f} tn={stats['tn']:.0f}"
        )
    if len(rows) >= 2:
        lo, hi = rows[0][1], rows[-1][1]
        span = f"{hi / lo:.1f}x" if lo else "undefined (p0=0 at the base level)"
        print(f"\np0 span across measured levels: {lo:.3f} -> {hi:.3f}  ({span})")
        print(
            "The sweep needs this span to be large: the cost of an always-FIX critique on the\n"
            "training signal is proportional to p0, so a wide span is what makes the\n"
            "pre-registered crossing observable rather than hypothetical."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
