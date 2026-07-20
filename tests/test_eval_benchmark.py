"""Tests for whystuck.eval.benchmark — task sets from traces or a tasks dir."""

from pathlib import Path

from whystuck.core.models import LoopTrace
from whystuck.eval.benchmark import benchmark_from_dir, benchmark_from_traces


def _trace(tid: str, task_id: str | None, summary: str) -> LoopTrace:
    return LoopTrace(
        id=tid,
        loop_id="bugfix",
        created_at="2026-07-20T00:00:00Z",
        task_id=task_id,
        task_summary=summary,
    )


def test_from_traces_dedupes_by_task():
    traces = [
        _trace("t1", "task-a", "fix login"),
        _trace("t2", "task-a", "fix login"),  # same task, second run
        _trace("t3", "task-b", "fix signup"),
    ]
    tasks = benchmark_from_traces(traces)
    assert {t.id for t in tasks} == {"task-a", "task-b"}


def test_from_traces_falls_back_to_trace_id_when_no_task_id():
    tasks = benchmark_from_traces([_trace("t1", None, "one-off")])
    assert tasks[0].id == "t1"
    assert tasks[0].payload["task_summary"] == "one-off"


def test_from_dir_loads_yaml_tasks(tmp_path: Path):
    (tmp_path / "task-001.yaml").write_text("id: task-001\npayload:\n  goal: fix the flaky test\n")
    (tmp_path / "task-002.yaml").write_text("id: task-002\npayload:\n  goal: speed up build\n")
    tasks = benchmark_from_dir(tmp_path)
    assert {t.id for t in tasks} == {"task-001", "task-002"}
    assert tasks[0].payload["goal"]


def test_from_dir_empty_when_missing(tmp_path: Path):
    assert benchmark_from_dir(tmp_path / "nope") == []
