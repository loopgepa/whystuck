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
