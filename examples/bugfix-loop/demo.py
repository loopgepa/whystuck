"""End-to-end WhyStuck demo — fully local, offline, no API key.

Run:  uv run python examples/bugfix-loop/demo.py

Shows the whole v0.1 flow against a temp store: record failing traces →
diagnose → propose a reversible ≤3-gene patch → dry-run promote (diff only).
Nothing is written outside the temp directories; nothing leaves your machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from whystuck.core.storage import WhyStuckStore
from whystuck.tools import (
    diagnose,
    promote,
    propose_fix,
    record_trace,
    save_loop,
)

LOOP = "bugfix-loop"


def _failing_trace(i: int) -> dict:
    return {
        "id": f"trace-{i}",
        "loop_id": LOOP,
        "created_at": "2026-07-20T00:00:00Z",
        "final_outcome": "failure",
        "failure_summary": "repair repeats the original patch after tests fail",
        "steps": [{"step_id": "verify", "role": "verify", "status": "failure"}],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as store_dir, tempfile.TemporaryDirectory() as proj_dir:
        store = WhyStuckStore(Path(store_dir))

        # 0. Register the loop and seed some failing runs (as a real session would).
        save_loop(
            store,
            {
                "id": LOOP,
                "name": "Bugfix Loop",
                "genes": [
                    {
                        "id": "verifier_rubric",
                        "type": "verifier_rubric",
                        "name": "Verifier Rubric",
                        "content": "Check whether the bug is fixed.",
                    }
                ],
            },
        )
        for i in range(8):
            record_trace(store, _failing_trace(i))

        # 1. Diagnose.
        modes = diagnose(store, LOOP)["failure_modes"]
        top = modes[0]
        print("Diagnosis")
        print(f"  failure mode : {top['title']}")
        print(f"  evidence     : {len(top['evidence_trace_ids'])} traces")
        print(f"  confidence   : {top['confidence']['band']} ({top['confidence']['score']})")
        print(f"  likely genes : {', '.join(top['likely_genes'])}")

        # 2. Propose a small, reversible, evidence-backed patch.
        proposal = {
            "id": "m1",
            "loop_id": LOOP,
            "mutation_type": "edit_gene",
            "target_gene_id": "verifier_rubric",
            "title": "Strengthen verifier rubric",
            "rationale": "Repairs keep passing a weak rubric; require minimal-patch evidence.",
            "evidence_trace_ids": top["evidence_trace_ids"],
            "after": (
                "Check whether the bug is fixed, the patch is minimal, and the failing "
                "test evidence directly supports the fix."
            ),
        }
        variant_id = propose_fix(store, LOOP, [proposal])["variant_id"]
        print(f"\nProposed variant: {variant_id}")

        # 3. Dry-run promote — show the diff, write nothing.
        dry = promote(store, LOOP, variant_id, base_dir=proj_dir, dry_run=True)
        print(f"\nDry-run promotion (dry_run={dry['dry_run']}, promoted={dry['promoted']})")
        print("  Approve in your agent to apply. Nothing has been written.")


if __name__ == "__main__":
    main()
