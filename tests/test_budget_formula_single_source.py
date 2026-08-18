"""The budget formulas the papers state must exist once, in the code that enforces them.

`accept_cycle_floor` decides whether a run is allowed to start: `run_validation_study` refuses a
budget below it. The same arithmetic was also written out in `verify_paper_claims`, as the expected
value for the column `tab:splitsweep` prints, and in the papers' prose. Two of those three copies
describe the decision without making it, so a change to GEPA's accounting would have left the
checker and the paper agreeing with each other and disagreeing with the harness. That is this
series' own subject arriving in its tooling, and the fix is the one the papers argue for: one
definition, and every other site calls it.
"""

from __future__ import annotations

import ast
import pathlib
import re

from whystuck.eval.study import accept_cycle_floor, promotable_cycles

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The arithmetic, as it would be written if someone inlined it again.
_FLOOR_ARITH = re.compile(r"2\s*\*\s*\w+(?:\.\w+)*\s*\+\s*min\(\s*3\s*,")
_CYCLES_ARITH = re.compile(r"\(\s*min\(\s*3\s*,\s*[\w.]+\s*\)\s*\+\s*[\w.]+\s*\)")


def _executable_lines(path: pathlib.Path) -> list[str]:
    """Source lines without docstrings or comments, so prose about the formula is not a copy."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.update(doc.splitlines())
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped in {d.strip() for d in docstrings}:
            continue
        out.append(line)
    return out


def test_the_floor_matches_the_columns_the_papers_print():
    """The four values `tab:splitsweep` prints, from the function the CLI enforces."""
    assert [accept_cycle_floor(n) for n in (4, 8, 16, 24)] == [11, 19, 35, 51]
    assert round(promotable_cycles(28, 8), 2) == 1.82


def test_the_arithmetic_is_written_once():
    hits = []
    for path in sorted(ROOT.glob("scripts/*.py")) + sorted(ROOT.glob("src/whystuck/**/*.py")):
        for i, line in enumerate(_executable_lines(path), 1):
            if _FLOOR_ARITH.search(line) or _CYCLES_ARITH.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:70]}")
    assert len(hits) == 2, (
        "the budget arithmetic should appear exactly twice in executable code, once per formula in "
        f"whystuck.eval.study; found {len(hits)}:\n  " + "\n  ".join(hits)
    )
    assert all("src/whystuck/eval/study.py" in h for h in hits), (
        "the arithmetic has been inlined outside the harness again:\n  " + "\n  ".join(hits)
    )


def test_every_caller_calls_the_shared_function():
    for rel, name in (
        ("scripts/run_validation_study.py", "accept_cycle_floor"),
        ("scripts/verify_paper_claims.py", "accept_cycle_floor"),
        ("scripts/verify_paper_claims.py", "promotable_cycles"),
        ("scripts/analyze_search_depth.py", "promotable_cycles"),
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert f"{name}(" in src, f"{rel} no longer calls {name}; it has its own copy again"
