#!/usr/bin/env python
"""Vendor a deterministic, sandbox-verified BigCodeBench-Hard subset (one-time dev tool).

BigCodeBench-Hard is hard even for frontier models (GPT-5-mini ~33% pass@1), so it gives
the optimization headroom MBPP+ lacked (which saturated — see the appendix). But its tasks
use scientific libraries and touch the filesystem / mock network, so scoring runs in a
pinned Docker sandbox (``whystuck-bcb``; see ``docker/bcb/Dockerfile``).

This keeps only tasks that are SOUND under our sandbox: the reference solution
(``complete_prompt`` + ``canonical_solution``) must pass the FULL unittest suite TWICE with
an identical verdict (deterministic — filters flaky/random/real-network tasks and any task
whose libs are absent from the image), run fast, and expose >= MIN_METHODS ``test_*`` methods
so the anti-oracle split has >= 2 visible and >= 2 hidden.

Per kept task we store: task_id, complete_prompt (the worker prompt), entry_point, the full
test source, the deterministic visible/hidden method split, the reference (for the pre-spend
preflight only; never shown), and libs. Written to
``src/whystuck/eval/data/bigcodebench_hard.json``.

    DOCKER_HOST=unix:///var/run/docker.sock \
      scratchpad/evalvenv/bin/python scripts/vendor_bigcodebench.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

REPO = "bigcode/bigcodebench-hard"
PARQUET = "data/v0.1.4-00000-of-00001.parquet"
OUT = Path("src/whystuck/eval/data/bigcodebench_hard.json")

MIN_METHODS = 4  # >= 2 visible + >= 2 hidden after the split
MAX_SECONDS = 12.0  # a reference run slower than this is dropped (keeps the loop tractable)
N_KEEP = 80  # cap the committed pool


def _test_methods(test_src: str) -> list[str]:
    """Sorted ``test_*`` method names of the first ``unittest.TestCase`` subclass."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                b.name
                for b in node.body
                if isinstance(b, ast.FunctionDef | ast.AsyncFunctionDef)
                and b.name.startswith("test")
            ]
            if methods:
                return sorted(methods)
    return []


def _split(methods: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic interleaved split: visible = even indices, hidden = odd. Interleaving
    (not first/second half) avoids grouping related consecutive tests on one side."""
    return methods[::2], methods[1::2]


def main() -> int:
    sys.path.insert(0, "src")
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    from whystuck.eval.bcb_exec import run_unittest

    rows = pq.read_table(hf_hub_download(REPO, PARQUET, repo_type="dataset")).to_pylist()
    rows.sort(key=lambda r: r["task_id"])

    kept: list[dict] = []
    reasons: dict[str, int] = {}

    def drop(why: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1

    for row in rows:
        if len(kept) >= N_KEEP:
            break
        methods = _test_methods(row["test"])
        if len(methods) < MIN_METHODS:
            drop("too_few_methods")
            continue
        reference = row["complete_prompt"] + row["canonical_solution"]
        t0 = time.monotonic()
        r1 = run_unittest(reference, row["test"], timeout=int(MAX_SECONDS) + 8)
        dt = time.monotonic() - t0
        if r1.verdict == "sandbox_error":
            drop("sandbox_error")
            continue
        if r1.verdict != "pass":
            drop(f"ref_{r1.verdict}")
            continue
        if dt > MAX_SECONDS:
            drop("too_slow")
            continue
        r2 = run_unittest(reference, row["test"], timeout=int(MAX_SECONDS) + 8)
        if r2.verdict != "pass":
            drop("nondeterministic")  # passed once, not twice
            continue
        visible, hidden = _split(methods)
        kept.append(
            {
                "task_id": row["task_id"],
                "complete_prompt": row["complete_prompt"],
                "entry_point": row["entry_point"],
                "test": row["test"],
                "visible_methods": visible,
                "hidden_methods": hidden,
                "reference": reference,
                "libs": row["libs"],
            }
        )
        print(f"[{len(kept):3d}] kept {row['task_id']} ({dt:.1f}s, {len(methods)} methods)")

    doc = {
        "_meta": {
            "source": f"huggingface.co/datasets/{REPO} ({PARQUET})",
            "license": "BigCodeBench: Apache-2.0",
            "image": "whystuck-bcb (docker/bcb/Dockerfile)",
            "description": "Deterministic, sandbox-verified BigCodeBench-Hard subset for the "
            "v2 code study. visible_methods = in-loop signal; hidden_methods = held-out scorer.",
            "n_tasks": len(kept),
            "scanned": len(rows),
            "drops": reasons,
            "filters": {
                "reference_passes_full_suite_twice": True,
                "min_test_methods": MIN_METHODS,
                "max_seconds": MAX_SECONDS,
                "split": "interleaved (visible=even idx, hidden=odd idx)",
            },
        },
        "tasks": kept,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nWrote {len(kept)} tasks to {OUT} ({OUT.stat().st_size // 1024} KB)")
    print("drops:", reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
