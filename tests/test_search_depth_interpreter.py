"""The depth measurement must run on an interpreter CI actually has.

WHY THIS EXISTS. `analyze_search_depth` shells out to a Python that can run the replay CLI, and
it named `scratchpad/evalvenv/bin/python` directly. `scratchpad/` is gitignored, so on a runner
that path does not exist and the subprocess raised FileNotFoundError -- during pytest COLLECTION,
because the claim checks are parametrized over `_checks()`. One missing gitignored directory took
the whole suite down, and main stayed red from 2026-07-31 until this was found on 2026-08-08
while trying to merge something unrelated.

The interpreter cannot affect the result: the measurement replays a committed cassette offline
and counts GEPA iteration lines. So a fallback is safe, and its absence was the only bug.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import analyze_search_depth as asd  # noqa: E402


def test_the_resolved_interpreter_exists():
    resolved = asd.eval_python()
    assert pathlib.Path(resolved).exists() or resolved == sys.executable, (
        f"{resolved!r} is neither a real path nor the running interpreter"
    )


def test_it_falls_back_when_no_venv_is_present(monkeypatch, tmp_path):
    # The CI condition: neither the eval venv nor the repo venv is on disk. It must resolve to
    # the running interpreter instead of returning a path that will raise on exec.
    monkeypatch.setattr(asd, "EVAL_PY", str(tmp_path / "absent" / "python"))
    monkeypatch.chdir(tmp_path)
    assert asd.eval_python() == sys.executable


def test_the_eval_venv_still_wins_when_it_is_there(monkeypatch, tmp_path):
    # Local runs must keep using the dedicated venv, so the fallback cannot silently change
    # which interpreter measures a published number on a developer machine.
    fake = tmp_path / "evalvenv" / "bin" / "python"
    fake.parent.mkdir(parents=True)
    fake.touch()
    monkeypatch.setattr(asd, "EVAL_PY", str(fake))
    assert asd.eval_python() == str(fake)
