"""Every number the paper prints in a table must be checked against an artifact.

WHY THIS TEST EXISTS. `verify_paper_claims.py` grew to hundreds of checks, but for most of its
life the majority of them compared a value computed from an artifact against a literal typed
into the script. That pins the ARTIFACT and never opens `main.tex`: edit a number in a table and
nothing fails. Two of this paper's published errors were of exactly that shape, and a third was
found by this test's diagnostic (a table row whose header marker was wrong, so six checks
silently vanished while the script still reported a pass).

So the property enforced here is not "the checks pass" -- that is `test_verify_paper_claims.py`
-- but "there is no printed table number that nothing checks". A new table, or a new column in
an old one, fails this test until it is either covered or deliberately exempted below.

The exemption list is deliberately small and each entry names why no artifact can produce the
number. Adding to it is a decision, not a shortcut.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import verify_paper_claims as vpc  # noqa: E402

PAPER = pathlib.Path("docs/paper/main.tex")

#: Same tokenizer `_table_row` uses, so coverage is measured over exactly the cells a check
#: could read. The lookbehind keeps LaTeX subscripts ($L_1$) from reading as data.
_NUM = re.compile(r"[+-]?\d+\.\d+|(?<![_\d])[+-]?\d+(?!\d)")

#: value -> why no artifact can produce it. Keys are rounded to 4dp, matching the comparison.
_EXEMPT: dict[float, str] = {
    160.0: (
        "tab:closability names mmc=160 as the budget a FUTURE replication would use (D4). The "
        "run was designed, started and stopped mid-flight when the audit landed; its partial "
        "cassette is deliberately not released, so there is no artifact to derive it from."
    ),
    3.7: (
        "tab:defects states the arithmetic identity 1/27 = 3.7% > alpha/2, which is why the "
        "exact-resample interval dropped its largest atom. It is a property of the estimator at "
        "n=3, not a measurement: 3 seeds give 3^3 = 27 resamples."
    ),
    27.0: "The 27 of that same 1/27. See 3.7.",
    -0.05: (
        "tab:ciexcl's 'as published' column is the OUTPUT OF THE BUG the table documents (a "
        "percentile-index error). No current code can reproduce it, by construction -- the fix "
        "removed the code path. The per-seed data and the corrected column ARE both checked."
    ),
    0.085: "The other endpoint of that same as-published interval. See -0.05.",
}


def _tables() -> list[tuple[str, list[float]]]:
    """(label, every number in the row body) for each tabular in the paper."""
    tex = PAPER.read_text(encoding="utf-8")
    out: list[tuple[str, list[float]]] = []
    for m in re.finditer(r"\\begin\{tabular\}", tex):
        end = tex.find(r"\bottomrule", m.end())
        if end < 0:
            continue
        label = re.search(r"\\label\{(tab:[^}]+)\}", tex[end : end + 2500])
        body = tex[tex.find(r"\midrule", m.end()) : end]
        out.append(
            (
                label.group(1) if label else "(unlabelled)",
                [round(float(v), 4) for v in _NUM.findall(body)],
            )
        )
    return out


def _checked_values() -> set[float]:
    return {round(float(c[1]), 4) for c in vpc._checks() if isinstance(c[1], (int, float))}


def test_every_table_number_is_checked_or_exempt():
    checked = _checked_values()
    unexplained: dict[str, list[float]] = {}
    for label, nums in _tables():
        missing = sorted({n for n in nums if n not in checked and n not in _EXEMPT})
        if missing:
            unexplained[label] = missing
    assert not unexplained, (
        "these table numbers are printed in main.tex and nothing checks them against an "
        f"artifact: {unexplained}. Either add a check to verify_paper_claims._checks() or add "
        "the value to _EXEMPT in this file with a reason no artifact can produce it."
    )


def test_the_exemption_list_does_not_rot():
    """An exemption for a value the paper no longer prints is dead weight that hides intent."""
    printed = {n for _, nums in _tables() for n in nums}
    stale = sorted(v for v in _EXEMPT if v not in printed)
    assert not stale, (
        f"_EXEMPT excuses {stale}, which the paper no longer prints in any table. Remove the "
        "entries so the list keeps meaning what it says."
    )


def test_coverage_is_not_vacuous():
    """The diagnostic must be looking at real tables and a real check set.

    Without this, a refactor that made `_tables()` return nothing would turn the test above into
    a tautology that passes forever -- the precise failure mode the module docstring describes.
    """
    tables = _tables()
    assert len(tables) >= 12, f"expected the paper's full set of tables, found {len(tables)}"
    assert sum(len(nums) for _, nums in tables) > 300, "table bodies look empty"
    assert len(_checked_values()) > 150, "the check set looks empty"


def test_table_row_joins_a_wrapped_row():
    """`tab:ciexcl` wraps its last column onto a continuation line. A line-based reader returns
    a short row, and a checker asking for that column then either raises or skips it."""
    tex = PAPER.read_text(encoding="utf-8")
    cells = vpc._table_row(
        tex, r"Result & per-seed & as published & corrected", "Actuator, powered"
    )
    assert len(cells) >= 4, "the corrected column was dropped, so the row was not joined"
    assert cells[3] == [-0.15, 0.0]
