"""The second worker's held-out tasks must be counted by content, not by id.

WHY THIS EXISTS. Paper II's register entry 29 said the twenty second-worker seeds "cover 62
distinct tasks, the most-reused appearing 8 times", and `_second_worker_pool_checks` pinned both
numbers. They were counts of task IDS. The multi-hard loader regenerates its pool per seed and
numbers tasks by position, so `mc-36` on seed 0 and `mc-36` on seed 5 are different tasks: the 200
held-out slots hold 200 distinct payloads. The check reproduced the paper's error because it keyed
on the same thing the paper did.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import verify_paper_claims as vpc  # noqa: E402


def test_second_worker_tasks_are_counted_by_payload():
    counts = vpc._second_worker_heldout_counts()
    # 20 seeds x 10 held-out tasks, each seed a fresh draw.
    assert counts.slots == 200
    # The id count is what the old check pinned; it is NOT the number of distinct tasks.
    assert counts.distinct_ids < counts.distinct_payloads
    assert counts.distinct_payloads == 200
    assert counts.max_payload_reuse == 1


def test_recorded_ids_are_the_loader_ids():
    # The payload count comes from regenerating the tasks, so the regenerated ids must be the
    # ones the released run recorded, seed by seed, or the count describes some other task set.
    assert vpc._second_worker_heldout_counts().seeds_matching_recorded_ids == 20


def test_pool_checks_pin_payloads_not_ids():
    checks = {label: actual for label, _p, _t, actual in vpc._second_worker_pool_checks()}
    assert checks["tex: second-worker distinct held-out tasks"] == 200.0
