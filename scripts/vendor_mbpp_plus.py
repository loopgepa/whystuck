#!/usr/bin/env python
"""Vendor a curated MBPP+ subset to a committed JSON (one-time, dev-only tool).

The v2 code study must replay bit-for-bit and run with no network, so the benchmark
data is *vendored*: this script downloads EvalPlus's ``mbppplus`` once, keeps only
tasks whose reference solution provably passes BOTH suites through our own sandbox
(``code_exec.run_tests``), and writes them to ``src/whystuck/eval/data/mbpp_plus.json``.
At study time ``code_bench`` reads only that file — pyarrow / huggingface_hub / the
network are needed here, never at run time.

Per task we store:
- ``task_id`` (int, MBPP id), ``prompt`` (NL description),
- ``entry_point`` (the function name the plus harness calls — the worker is told to
  define it),
- ``base_tests`` (the visible ``test_list`` asserts, with ``test_imports`` prepended)
  — the in-loop signal,
- ``plus_test`` (the hidden edge-case harness, verbatim) — the held-out scorer,
- ``reference`` (the canonical solution) — kept ONLY to re-verify satisfiability; never
  shown to the worker.

Filters (all enforced before a task is kept, so no unsatisfiable/slow/huge task lands
in the committed set): the reference passes base AND plus; the entry point is extractable
and defined by the reference; the plus harness is lean (≤ MAX_PLUS_CHARS) and fast
(≤ PLUS_TIMEOUT s). Tasks are taken in ``task_id`` order (deterministic) up to N_KEEP.

    scratchpad/evalvenv/bin/python scripts/vendor_mbpp_plus.py   # needs numpy+pyarrow

Difficulty calibration (which subset is actually hard for the worker) is a *separate*,
pre-spend pilot step (see the pre-registration) — this script only guarantees the data
is valid and reproducible, not that it has headroom for a given model.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = "evalplus/mbppplus"
PARQUET = "data/test-00000-of-00001-d5781c9c51e02795.parquet"
OUT = Path("src/whystuck/eval/data/mbpp_plus.json")

N_KEEP = 64  # committed pool size; load_code_tasks splits it into disjoint train/heldout
MAX_PLUS_CHARS = 6000  # keep the vendored file lean and the sandbox run fast
PLUS_TIMEOUT = 3.0  # a reference plus-run slower than this is dropped (runaway guard)

# The plus harness ends with `assertion(<entry_point>(*inp), exp, ...)`; grab that name.
_ENTRY_RE = re.compile(r"assertion\(\s*([A-Za-z_]\w*)\s*\(")


def _entry_point(plus_test: str, reference: str) -> str | None:
    """The function name the plus harness calls, iff the reference defines it."""
    hits = _ENTRY_RE.findall(plus_test)
    if not hits:
        return None
    name = hits[-1]  # the final assertion loop calls the entry point
    return name if re.search(rf"\bdef\s+{re.escape(name)}\s*\(", reference) else None


def main() -> int:
    sys.path.insert(0, "src")
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    from whystuck.eval.code_exec import run_tests

    rows = pq.read_table(hf_hub_download(REPO, PARQUET, repo_type="dataset")).to_pylist()
    rows.sort(key=lambda r: int(r["task_id"]))

    kept: list[dict] = []
    scanned = 0
    for row in rows:
        if len(kept) >= N_KEEP:
            break
        scanned += 1
        plus_test = row["test"]
        reference = row["code"]
        if len(plus_test) > MAX_PLUS_CHARS:
            continue
        entry = _entry_point(plus_test, reference)
        if entry is None:
            continue
        base_tests = list(row["test_imports"]) + list(row["test_list"])
        base_block = "\n".join(base_tests)

        rb = run_tests(reference, base_block, timeout=10.0)
        if not rb.passed:
            continue
        t0 = time.monotonic()
        rp = run_tests(reference, plus_test, timeout=10.0)
        if not rp.passed or (time.monotonic() - t0) > PLUS_TIMEOUT:
            continue

        kept.append(
            {
                "task_id": int(row["task_id"]),
                "prompt": row["prompt"],
                "entry_point": entry,
                "base_tests": base_tests,
                "plus_test": plus_test,
                "reference": reference,
            }
        )

    if len(kept) < N_KEEP:
        print(
            f"WARNING: only {len(kept)} tasks passed the filters (wanted {N_KEEP}); "
            f"scanned {scanned} rows.",
            file=sys.stderr,
        )

    doc = {
        "_meta": {
            "source": f"huggingface.co/datasets/{REPO} ({PARQUET})",
            "license": "MBPP: CC-BY-4.0; EvalPlus plus tests: Apache-2.0",
            "description": "Curated, reference-verified MBPP+ subset for the WhyStuck v2 "
            "code study. base_tests = visible/in-loop; plus_test = hidden/held-out scorer.",
            "n_tasks": len(kept),
            "filters": {
                "reference_passes_base_and_plus": True,
                "max_plus_chars": MAX_PLUS_CHARS,
                "plus_timeout_s": PLUS_TIMEOUT,
                "order": "ascending task_id, first N_KEEP",
            },
        },
        "tasks": kept,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"Wrote {len(kept)} tasks to {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
