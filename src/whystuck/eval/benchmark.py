"""Build a benchmark task set — by replaying recorded traces, or from a tasks dir.

WhyStuck never fabricates a synthetic suite: a "benchmark" is either the tasks
that actually ran (replay) or ones you declared (ADR-011 / v0.2 design).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML

from whystuck.core.models import LoopTrace

_yaml = YAML(typ="safe")


class BenchmarkTask(BaseModel):
    id: str
    payload: dict[str, Any] = Field(default_factory=dict)


#: Task sets whose loader REGENERATES its pool per seed, so two seeds are independent
#: draws and a statistic computed across seeds has the replication it assumes.
#:
#: The distinction is invisible in the loaders — all of them end in
#: `shuffle(pool); split at mid` — but `load_multi_constraint` calls
#: `generate_multi_tasks(pool, seed=seed)` FIRST, reseeding the pool itself, while the
#: vendored loaders shuffle a fixed file. There, "seed" only chooses a partition of the
#: same tasks, so seeds share about half their held-out split. Measured pairwise held-out
#: overlap: ifbench 15.4/31 (50%), bcb 20.3/40 (51%), code 16/32 (50%), and 0/32 for
#: multi and multi-hard (identical ids, zero identical payloads).
#:
#: This is load-bearing for inference, not bookkeeping: per-seed differences from a
#: shared pool are positively correlated, which makes a bootstrap interval over seeds too
#: narrow and a sign test's p too small. Both err toward overstating significance, so a
#: statistic over seeds of a vendored task set cannot be read as replication.
SEED_INDEPENDENT_TASK_SETS = frozenset({"multi", "multi-hard"})

#: Every task set the study can run, so an unknown name fails loud rather than silently
#: defaulting to "independent" (the permissive answer) in the analysis.
KNOWN_TASK_SETS = frozenset({"ifbench", "multi", "multi-hard", "code", "bcb"})


def seeds_are_independent(task_set: str) -> bool:
    """Whether different seeds of `task_set` are independent draws (see above)."""
    if task_set not in KNOWN_TASK_SETS:
        raise ValueError(
            f"unknown task set {task_set!r}; add it to KNOWN_TASK_SETS and decide whether "
            "its loader regenerates the pool per seed — defaulting would silently grant "
            "replication a new task set may not have"
        )
    return task_set in SEED_INDEPENDENT_TASK_SETS


def benchmark_from_traces(traces: list[LoopTrace]) -> list[BenchmarkTask]:
    """One task per distinct task_id (fallback: trace id), preserving first-seen order."""
    seen: OrderedDict[str, BenchmarkTask] = OrderedDict()
    for trace in traces:
        task_id = trace.task_id or trace.id
        if task_id not in seen:
            seen[task_id] = BenchmarkTask(id=task_id, payload={"task_summary": trace.task_summary})
    return list(seen.values())


def benchmark_from_dir(path: Path) -> list[BenchmarkTask]:
    """Load declared task specs (*.yaml / *.json) from a directory."""
    if not path.exists():
        return []
    tasks: list[BenchmarkTask] = []
    for f in sorted([*path.glob("*.yaml"), *path.glob("*.yml"), *path.glob("*.json")]):
        try:
            tasks.append(BenchmarkTask.model_validate(_yaml.load(f.read_text(encoding="utf-8"))))
        except ValidationError as exc:
            # Name the culprit — a bare pydantic error over a whole tasks/ dir is
            # unsearchable when one file among many is malformed.
            raise ValueError(f"invalid benchmark task file {f.name}: {exc}") from exc
    return tasks
