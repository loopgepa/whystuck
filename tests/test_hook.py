"""Tests for the zero-dep PostToolUse thrash hook (plugin/hooks/thrash_watch.py)."""

import json
import subprocess
import sys
from pathlib import Path

from whystuck.core.storage import WhyStuckStore
from whystuck.tools import diagnose

_HOOK = Path(__file__).resolve().parents[1] / "plugin" / "hooks" / "thrash_watch.py"


def _event(file_path: str, tool: str = "Edit", session: str = "s1") -> dict:
    return {
        "session_id": session,
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": {"file_path": file_path},
        "tool_response": {"success": True},
    }


def _run(event: dict, home: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env={**_base_env(), "WHYSTUCK_HOME": str(home)},
    )
    assert proc.returncode == 0
    return proc.stdout


def _base_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items()}


def test_no_nudge_before_threshold(tmp_path: Path):
    for _ in range(3):  # threshold is 4
        out = _run(_event("/proj/app.py"), tmp_path)
        assert out.strip() == ""
    assert not (tmp_path / "traces" / "session.jsonl").exists()


def test_nudge_and_trace_on_threshold(tmp_path: Path):
    out = ""
    for _ in range(4):
        out = _run(_event("/proj/app.py"), tmp_path)
    directive = json.loads(out)
    assert directive["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "whystuck_diagnose" in directive["hookSpecificOutput"]["additionalContext"]
    assert (tmp_path / "traces" / "session.jsonl").read_text().count("\n") == 1


def test_nudge_is_once_per_file(tmp_path: Path):
    outs = [_run(_event("/proj/app.py"), tmp_path) for _ in range(6)]
    nudges = [o for o in outs if o.strip()]
    assert len(nudges) == 1  # only the 4th edit nudges, not the 5th/6th


def test_non_edit_tools_ignored(tmp_path: Path):
    for _ in range(5):
        assert _run(_event("/proj/app.py", tool="Bash"), tmp_path).strip() == ""


def test_malformed_input_exits_clean(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input="not json",
        capture_output=True,
        text=True,
        env={**_base_env(), "WHYSTUCK_HOME": str(tmp_path)},
    )
    assert proc.returncode == 0


def test_captured_trace_feeds_diagnose(tmp_path: Path):
    for _ in range(4):
        _run(_event("/proj/app.py"), tmp_path)
    result = diagnose(WhyStuckStore(tmp_path), "session")
    assert result["trace_count"] == 1
    assert "repeated edits" in result["failure_modes"][0]["title"]
