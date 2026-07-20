#!/usr/bin/env python3
"""WhyStuck PostToolUse hook — detect loop *thrash* and nudge the agent to diagnose.

Zero-dependency (stdlib only) so it runs under plain python3 without WhyStuck
installed, and fail-open (any error → silent exit 0, never breaks the session).

Signal: the same file edited many times in one session without it sticking — the
canonical "edit the same function, same failure, edit again" thrash. On the Nth
edit it (a) appends a session trace to WHYSTUCK_HOME so `whystuck_diagnose` has
real data, and (b) emits an additionalContext nudge to the agent — once per file.

Reads the PostToolUse event JSON on stdin; may print a JSON directive on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_EDIT_TOOLS = {"Write", "Edit", "MultiEdit"}
_SESSION_LOOP = "session"


def _home() -> Path:
    if h := os.environ.get("WHYSTUCK_HOME"):
        return Path(h)
    if x := os.environ.get("XDG_DATA_HOME"):
        return Path(x) / "whystuck"
    return Path.home() / ".local" / "share" / "whystuck"


def _threshold() -> int:
    try:
        return max(2, int(os.environ.get("WHYSTUCK_THRASH_EDITS", "4")))
    except ValueError:
        return 4


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


def main() -> None:
    raw = sys.stdin.read()
    event = json.loads(raw)
    if event.get("tool_name") not in _EDIT_TOOLS:
        return
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return

    home = _home()
    session_id = str(event.get("session_id") or "unknown")
    state_path = home / "hookstate" / f"{session_id}.json"
    state = {"edits": {}, "nudged": []}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass

    edits = state.setdefault("edits", {})
    nudged = state.setdefault("nudged", [])
    edits[file_path] = int(edits.get(file_path, 0)) + 1

    threshold = _threshold()
    tripped = edits[file_path] >= threshold and file_path not in nudged

    if tripped:
        nudged.append(file_path)
        summary = f"repeated edits to {file_path} ({edits[file_path]}x) without resolution"
        _append_jsonl(
            home / "traces" / f"{_SESSION_LOOP}.jsonl",
            {
                "id": f"session-{uuid.uuid4().hex[:8]}",
                "loop_id": _SESSION_LOOP,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "final_outcome": "failure",
                "failure_summary": summary,
                "steps": [{"step_id": "edit", "role": "edit", "status": "failure"}],
            },
        )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    if tripped:
        nudge = (
            f"WhyStuck: {file_path} has been edited {edits[file_path]} times this "
            "session without sticking — a likely stuck loop. Before editing it again, "
            "run whystuck_diagnose on the 'session' loop (or /whystuck:explain-loop-failure "
            "session) to see the recurring failure and a reversible fix."
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": nudge,
                    }
                }
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - fail open: never break the user's session
        sys.exit(0)
