"""Shared subprocess capture for the v0.1 runner and the v0.2 eval repo worker.

One place decides how a worker/verifier command is executed and how its output is
summarized, so both call sites stay byte-for-byte consistent (they had drifted:
one head-capped with an ellipsis, the other raw-sliced).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Cap stored command output so we keep a useful summary, not raw logs (ADR-004.1).
OUTPUT_SUMMARY_CAP = 500
# Default wall-clock cap for a single command; a hung command can't hang the tool.
DEFAULT_TIMEOUT_S = 300


def run_capture(
    cmd: list[str], *, cwd: Path | None = None, timeout: int = DEFAULT_TIMEOUT_S
) -> tuple[int, str]:
    """Run `cmd`, returning (exit_code, combined_output).

    stderr precedes stdout so the real error survives the summary head-cap;
    errors="replace" means non-UTF-8 output never crashes the run.
    """
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stderr + proc.stdout)


def summarize(output: str) -> str:
    """Strip and head-cap command output for storage (keeps a summary, not logs)."""
    output = output.strip()
    if len(output) <= OUTPUT_SUMMARY_CAP:
        return output
    return output[:OUTPUT_SUMMARY_CAP] + "…"
