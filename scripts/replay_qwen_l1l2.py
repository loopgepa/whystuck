#!/usr/bin/env python
"""Replay the third-worker L1/L2 test end to end and measure what it cost. Zero cost, offline.

Two checks the registered analysis (`analyze_qwen_l1l2.py`) does not make, because it replays only
the held-out scoring of the released genomes:

  1. **The whole run reproduces.** Each level is re-run under `--replay` from its committed
     cassette, the same arguments the recording used, and every arm's `seed_heldout`,
     `opt_heldout`, `genes_changed`, `final_genes` and `train_score` is compared with the released
     JSON. A cassette miss raises inside the runner, so a replay that exits 0 made no live call.
  2. **What it cost, as a floor.** The replay is instrumented the way `measure_actual_spend.py`
     instruments its registry runs: billing is one call per distinct (surface, prompt), tokens are
     counted with `cl100k_base`, and the same three caveats hold (a failed and retried call left no
     entry, the tokenizer is not the provider's, and any reasoning tokens the provider billed but
     did not return are not in the cassette). The L2 recording also lost three attempts to
     provider timeouts; calls in flight when an attempt died are not in the cassette either.

Writes `docs/qwen-l1l2/replay.json`. Run with the eval venv (it drives GEPA):

    PYTHONPATH=src scratchpad/evalvenv/bin/python scripts/replay_qwen_l1l2.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import measure_actual_spend as mas  # noqa: E402
import run_validation_study as cli  # noqa: E402

from whystuck.eval.cassette import Cassette  # noqa: E402

RESULTS = "docs/qwen-l1l2/validation-study-qwen-{level}.json"
CASSETTE = "tests/fixtures/cassettes/study-qwen-l1l2-{level}.json"
OUT = Path("docs/qwen-l1l2/replay.json")
FIELDS = ("seed_heldout", "opt_heldout", "genes_changed", "final_genes", "train_score")
# The recording's arguments (pre-registration, "Design"), less --max-usd, --yes and the paths.
ARGS = [
    "--task-set", "multi-hard", "--arms", "ABCD", "--seeds", "0,1,2,3,4,5,6,7,8,9",
    "--n-train", "8", "--n-heldout", "10", "--max-metric-calls", "28",
    "--reflection-calls", "6", "--module-selector", "all", "--loop-max-retries", "1",
    "--worker-model", "groq/qwen/qwen3.8-27b", "--reflection-model", "groq/openai/gpt-oss-120b",
    "--empty-attempts", "6",
]  # fmt: skip
LEVEL_ARGS = {"L1": [], "L2": ["--exclude-constraint", "exact_word_count"]}


def _replay(level: str, out_dir: Path) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    """Replay one level, returning the replayed result JSON and every distinct call it served."""
    seen: dict[tuple[str, str], str] = {}
    original = Cassette.wrap

    def spy(self: Cassette, fn: Callable[..., str], *, name: str) -> Callable[..., str]:
        inner = original(self, fn, name=name)

        def recorded(*a: Any, **kw: Any) -> str:
            output = inner(*a, **kw)
            prompt = a[0] if a and isinstance(a[0], str) else json.dumps([a, sorted(kw)])
            seen.setdefault((name, prompt), output)
            return output

        return recorded

    out = out_dir / f"{level}.json"
    Cassette.wrap = spy  # type: ignore[method-assign]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = cli.main(
                [
                    "--replay",
                    *ARGS,
                    *LEVEL_ARGS[level],
                    "--record",
                    CASSETTE.format(level=level),
                    "--out",
                    str(out),
                    "--report",
                    str(out_dir / f"{level}.md"),
                    "--max-usd",
                    "99",
                ]
            )
    finally:
        Cassette.wrap = original  # type: ignore[method-assign]
    if code != 0:
        raise SystemExit(f"{level}: replay exited {code}; output:\n{buf.getvalue()[-2000:]}")
    return json.loads(out.read_text(encoding="utf-8")), seen


def main() -> int:  # pragma: no cover - needs the released runs and the eval venv
    report: dict[str, Any] = {}
    count = mas._encoder()
    with tempfile.TemporaryDirectory() as d:
        for level in LEVEL_ARGS:
            replayed, seen = _replay(level, Path(d))
            released = json.loads(Path(RESULTS.format(level=level)).read_text(encoding="utf-8"))
            mismatches = [
                f"seed {i} arm {arm} {field}"
                for i, (a, b) in enumerate(zip(released["runs"], replayed["runs"], strict=True))
                for arm in "ABCD"
                for field in FIELDS
                if a[arm].get(field) != b[arm].get(field)
            ]
            compared = len(released["runs"]) * 4 * len(FIELDS)
            surfaces: dict[str, dict[str, int]] = defaultdict(
                lambda: {"calls": 0, "tin": 0, "tout": 0}
            )
            for (name, prompt), output in seen.items():
                s = surfaces[name]
                s["calls"] += 1
                s["tin"] += count(prompt)
                s["tout"] += count(output or "")
            entries = json.loads(Path(CASSETTE.format(level=level)).read_text(encoding="utf-8"))
            usd = mas._price(dict(surfaces), mas._surface_models(ARGS))
            report[level] = {
                "fields_compared": compared,
                "mismatches": mismatches,
                "cassette_entries": len(entries["entries"]),
                "distinct_calls_served": len(seen),
                "surfaces": dict(surfaces),
                "usd_floor": sum(usd.values()),
                "bound_usd": released["meta"]["estimated_usd"],
            }
            print(
                f"{level}: {compared} fields compared, {len(mismatches)} mismatches, "
                f"{len(seen)} distinct calls of {len(entries['entries'])} cassette entries, "
                f"floor ${sum(usd.values()):.4f} (bound ${released['meta']['estimated_usd']:.4f})"
            )
    report["usd_floor_total"] = sum(report[lv]["usd_floor"] for lv in LEVEL_ARGS)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"total floor ${report['usd_floor_total']:.4f}; see {OUT}")
    bad = [lv for lv in LEVEL_ARGS if report[lv]["mismatches"]]
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
