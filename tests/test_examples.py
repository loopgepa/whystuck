"""Regression guard for the offline example scripts.

The README's headline "try it offline" command runs these scripts. Tests
elsewhere never import ``examples/``, so a rename in the tool surface can break
the demos while the suite stays green. Executing each script here closes that
gap: an ImportError or a runtime failure fails a test instead of only the
user's first run.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "bugfix-loop"


@pytest.mark.parametrize("script", ["demo.py", "demo_full.py"])
def test_example_script_runs_end_to_end(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
    out = capsys.readouterr().out
    assert "Diagnosis" in out
