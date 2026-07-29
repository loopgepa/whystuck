#!/usr/bin/env python
"""Measure the search depth each budget-sweep run actually achieved (offline, $0).

WHY THIS EXISTS. The pre-registration priced search depth as
``(mmc - n_train) / (min(3, n_train) + n_train)`` -- promotable accept cycles -- on the
assumption that every GEPA proposal costs a reflection minibatch PLUS a full valset
re-evaluation. That is only true of a proposal that WINS on the minibatch: a rejected one costs
the minibatch alone. So the formula prices the expensive case and badly understates how many
iterations a budget buys. Measured, the proposal cap of 6 binds from about ``mmc=40``, not from
``mmc=74`` as registered, which is why three cells of the sweep landed at the same depth.

The honest fix is to stop computing depth and start measuring it. GEPA logs one
``Iteration N:`` line per proposal round, so replaying a run from its cassette recovers the
count exactly, for free, with no API key. That makes the paper's depth column a derived
quantity rather than four numbers transcribed from a terminal.

    python scripts/analyze_search_depth.py
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

EVAL_PY = "scratchpad/evalvenv/bin/python"
CLI = "scripts/run_validation_study.py"
CASSETTES = "tests/fixtures/cassettes"

#: (label, cassette slug, mmc, proposal cap). n_train is 8 and every other lever is the
#: four-arm re-run's, so mmc and the cap are the only things that move.
BUDGET_RUNS: tuple[tuple[str, str, int, int], ...] = (
    ("mmc=20", "mmc20", 20, 6),
    ("mmc=28", "c2b", 28, 6),
    ("mmc=40", "mmc40", 40, 6),
    ("mmc=56", "mmc56", 56, 6),
    ("mmc=80", "mmc80", 80, 6),
    ("mmc=160", "mmc160", 160, 12),
)

_ITER = re.compile(r"Iteration (\d+):")


def priced_cycles(mmc: int, n_train: int = 8) -> float:
    """The pre-registered estimate, kept so the paper can show how wrong it was."""
    return (mmc - n_train) / (min(3, n_train) + n_train)


def measured_depth(slug: str, mmc: int, cap: int) -> int | None:
    """Highest GEPA iteration index reached, replayed from the cassette. None if absent.

    Replay rather than record: no key, no spend, and a cassette miss raises rather than
    silently truncating the count -- a depth read off a partial replay would understate the
    search and flatter the "budget does not help" reading.
    """
    cassette = Path(f"{CASSETTES}/study-{slug}.json")
    if not cassette.exists():
        return None
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(
            [
                EVAL_PY,
                CLI,
                "--task-set",
                "multi-hard",
                "--arms",
                "BD",
                "--seeds",
                "0,1,2",
                "--n-train",
                "8",
                "--n-heldout",
                "10",
                "--max-metric-calls",
                str(mmc),
                "--reflection-calls",
                str(cap),
                "--loop-max-retries",
                "1",
                "--module-selector",
                "all",
                "--reflection-model",
                "openai/gpt-5.6-luna",
                "--replay",
                "--record",
                str(cassette),
                "--out",
                str(Path(d) / "out.json"),
            ],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        return None
    found = [int(m) for m in _ITER.findall(proc.stdout)]
    return max(found) if found else None


def main() -> int:  # pragma: no cover - CLI convenience
    print("=== search depth: priced vs measured ===")
    print("priced = the pre-registered promotable-cycle formula; measured = GEPA iterations")
    print("actually reached, recovered by replaying each cassette offline.\n")
    print(f"{'run':9} {'cap':>4} {'priced':>7} {'measured':>9}  binding")
    for label, slug, mmc, cap in BUDGET_RUNS:
        depth = measured_depth(slug, mmc, cap)
        priced = priced_cycles(mmc)
        if depth is None:
            print(f"{label:9} {cap:>4} {priced:>7.2f} {'--':>9}  not bought / not replayable")
            continue
        binding = "proposal cap" if depth >= cap else "budget"
        print(f"{label:9} {cap:>4} {priced:>7.2f} {depth:>9}  {binding}")
    print(
        "\nThe formula overstates the cost of a proposal, because a proposal rejected on the\n"
        "reflection minibatch never pays for the full valset re-evaluation. So the cap binds\n"
        "earlier than priced, and cells above that point share one search depth."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
