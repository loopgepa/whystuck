"""End-to-end pipeline test: record → diagnose → propose → dry-run → promote.

Ties every core module together through the tool layer, offline, in temp dirs.
This is the flow the bugfix-loop example and the demo GIF walk through.
"""

from pathlib import Path

from whystuck.core.models import ConfidenceBand
from whystuck.core.storage import WhyStuckStore
from whystuck.tools import (
    diagnose,
    promote_variant,
    propose_mutations,
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


def test_full_diagnose_propose_promote_flow(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    (proj / "skills" / "bugfix").mkdir(parents=True)
    (proj / "skills" / "bugfix" / "SKILL.md").write_text("Fix the bug.")

    # A loop with a file-backed skill gene we will patch.
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
                    "path": "skills/bugfix/SKILL.md",
                    "content": "Fix the bug. Prefer a minimal patch and show the failing test.",
                }
            ],
        },
    )

    # 15 failing runs, all the same mode → HIGH confidence, verifier implicated.
    for i in range(15):
        record_trace(store, _failing_trace(i))
    top = diagnose(store, LOOP)["failure_modes"][0]
    assert top["confidence"]["band"] == ConfidenceBand.HIGH
    assert len(top["evidence_trace_ids"]) == 15
    assert "verifier_rubric" in top["likely_genes"]

    # Propose a patch to the skill gene, backed by the evidence.
    variant_id = propose_mutations(
        store,
        LOOP,
        [
            {
                "id": "m1",
                "loop_id": LOOP,
                "mutation_type": "edit_gene",
                "target_gene_id": "bugfix_skill",
                "title": "Require minimal-patch evidence",
                "rationale": "Repairs keep repeating; force minimal-patch justification.",
                "evidence_trace_ids": top["evidence_trace_ids"],
                "after": "Fix the bug with the smallest patch and cite the failing test.",
            }
        ],
    )["variant_id"]

    # Dry-run writes nothing.
    dry = promote_variant(store, LOOP, variant_id, base_dir=str(proj), dry_run=True)
    assert dry["dry_run"] is True
    assert (proj / "skills" / "bugfix" / "SKILL.md").read_text() == "Fix the bug."

    # Approved promotion writes the file and records it reversibly.
    live = promote_variant(store, LOOP, variant_id, base_dir=str(proj), dry_run=False)
    assert live["promoted"] is True
    assert "smallest patch" in (proj / "skills" / "bugfix" / "SKILL.md").read_text()
    assert len(store.list_promotions(LOOP)) == 1
