"""Full LoopGEPA flow for the demo GIF — diagnose → propose → APPROVE → promote.

Run:  uv run python examples/bugfix-loop/demo_full.py

Shows the write actually happening, atomically and reversibly, against a throwaway
project. Offline, no API key. Nothing outside the temp dirs is touched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from loopgepa.core.storage import LoopGEPAStore
from loopgepa.tools import infer, promote_variant, propose_mutations, record_trace, save_loop

LOOP = "bugfix-loop"
SKILL_REL = "skills/bugfix/SKILL.md"


def _failing(i: int) -> dict:
    return {
        "id": f"trace-{i}",
        "loop_id": LOOP,
        "created_at": "2026-07-20T00:00:00Z",
        "final_outcome": "failure",
        "failure_summary": "repair repeats the original patch after tests fail",
        "steps": [{"step_id": "verify", "role": "verify", "status": "failure"}],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as store_dir, tempfile.TemporaryDirectory() as proj:
        store = LoopGEPAStore(Path(store_dir))
        skill = Path(proj) / SKILL_REL
        skill.parent.mkdir(parents=True)
        skill.write_text("Fix the bug.")

        save_loop(
            store,
            {
                "id": LOOP,
                "name": "Bugfix Loop",
                "genes": [
                    {
                        "id": "bugfix_skill",
                        "type": "skill",
                        "name": "Bugfix Skill",
                        "path": SKILL_REL,
                        "content": "Fix the bug with the smallest patch and cite the failing test.",
                    }
                ],
            },
        )
        for i in range(12):
            record_trace(store, _failing(i))

        top = infer(store, LOOP)["failure_modes"][0]
        print(f"Diagnosis: {top['title']}")
        print(f"  {len(top['evidence_trace_ids'])} traces · confidence {top['confidence']['band']}")

        vid = propose_mutations(
            store,
            LOOP,
            [
                {
                    "id": "m1",
                    "loop_id": LOOP,
                    "mutation_type": "edit_gene",
                    "target_gene_id": "bugfix_skill",
                    "title": "Require minimal-patch + failing-test evidence",
                    "rationale": "Repairs keep repeating; force a minimal, evidenced patch.",
                    "evidence_trace_ids": top["evidence_trace_ids"],
                    "after": "Fix the bug with the smallest patch and cite the failing test.",
                }
            ],
        )["variant_id"]
        print(f"\nProposed {vid} — dry-run diff:")
        dry = promote_variant(store, LOOP, vid, base_dir=proj, dry_run=True)
        for line in dry["diffs"][0].splitlines()[2:6]:
            print(f"  {line}")

        print(f"\nBEFORE:  {skill.read_text()!r}")
        promote_variant(store, LOOP, vid, base_dir=proj, approved_by="human", dry_run=False)
        print(f"AFTER :  {skill.read_text()!r}")
        rec = store.list_promotions(LOOP)[-1]
        print(
            f"\nPromoted · reversible · rollback record: {rec.files_changed} by {rec.approved_by}"
        )


if __name__ == "__main__":
    main()
