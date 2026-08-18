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

import paper_sources  # noqa: E402
import verify_paper_claims as vpc  # noqa: E402

# The three papers, concatenated with their shared inputs expanded. Reading them through
# `paper_sources` rather than one hard-coded path is what keeps this guard covering the whole series
# after the 46-page draft was split three ways: a guard that silently narrowed its own scope is
# precisely the defect these papers are about.
PAPERS_TEX = paper_sources.all_tex

#: Same tokenizer `_table_row` uses, so coverage is measured over exactly the cells a check
#: could read. The lookbehind keeps LaTeX subscripts ($L_1$) from reading as data.
_NUM = re.compile(r"[+-]?\d+\.\d+|(?<![_\d])[+-]?\d+(?!\d)")

#: A model identifier is a name, not a measurement, and `llama-3.3-70b` tokenizes as -3.3 and 70
#: while `gpt-4.1-mini` tokenizes as -4.1. Demanding an artifact for those would push a tester
#: toward exempting real values to silence the noise, which is the one way this guard can be made
#: worse. Every `\texttt` span in the three papers holds an identifier, a path or quoted verbatim
#: output -- no data number lives inside one -- so the whole span is dropped before tokenizing.
_VERBATIM = re.compile(r"\\texttt\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

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
    tex = PAPERS_TEX()
    out: list[tuple[str, list[float]]] = []
    for m in re.finditer(r"\\begin\{tabular\}", tex):
        end = tex.find(r"\bottomrule", m.end())
        if end < 0:
            continue
        label = re.search(r"\\label\{(tab:[^}]+)\}", tex[end : end + 2500])
        body = _VERBATIM.sub(" ", tex[tex.find(r"\midrule", m.end()) : end])
        out.append(
            (
                label.group(1) if label else "(unlabelled)",
                [round(float(v), 4) for v in _NUM.findall(body)],
            )
        )
    return out


def test_no_exemption_has_gone_dead():
    """Every entry in `_EXEMPT` must still name a number some table prints.

    A dead exemption is a silent widening: it stops describing the paper and starts standing
    ready to excuse whatever value happens to collide with it next. This caught nothing when
    written -- it exists so that narrowing the tokenizer, which is how `_VERBATIM` arrived,
    cannot orphan an entry without saying so.
    """
    printed = {n for _, nums in _tables() for n in nums}
    dead = sorted(value for value in _EXEMPT if value not in printed)
    assert not dead, (
        f"these exemptions no longer match any printed table number: {dead}. Delete them, or "
        "find out why the value stopped being printed."
    )


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
    tex = PAPERS_TEX()
    cells = vpc._table_row(
        tex, r"Result & per-seed & as published & corrected", "Actuator, powered"
    )
    assert len(cells) >= 4, "the corrected column was dropped, so the row was not joined"
    assert cells[3] == [-0.15, 0.0]


def test_no_table_cell_holds_a_bare_comma() -> None:
    r"""An empty cell in these tables is written `---`, and a comma there is edit damage.

    The prose pass of 2026-08-17 replaced every em dash used as punctuation, and its rule for
    `--- &` fired on nine cells where the dash was a *placeholder* rather than punctuation: the
    baseline row of `tab:secondworker` came out as `0.7895 &, &, &,`. Nothing else caught it. The
    table-coverage guard above reads numbers, a comma is not a number, and no check reads a cell
    that is supposed to be empty.

    A placeholder is typography, not prose, so the writing rule does not reach it and this test
    says so in one place instead of leaving the next pass to rediscover it.
    """
    import re

    import paper_sources

    damaged = []
    for slug in paper_sources.PAPERS:
        for i, line in enumerate(paper_sources.paper_tex(slug).splitlines(), 1):
            if line.lstrip().startswith("%") or "&" not in line:
                continue
            for cell in line.split("&"):
                if re.fullmatch(r"\s*,\s*(\\\\)?\s*", cell):
                    damaged.append(f"{slug}:{i}: {line.strip()[:80]}")
    assert not damaged, (
        "table cells holding a bare comma where a placeholder belongs:\n  " + "\n  ".join(damaged)
    )
