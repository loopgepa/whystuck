"""Every value the paper prints in units of sigma must be one an artifact produces.

WHY THIS TEST EXISTS. `test_paper_table_coverage.py` closed the same hole for tables; this one
closes it for the prose. The paper restates a handful of gaps in sigma units across the abstract,
the contributions list, two table captions, the results section, the threats section and the
conclusion --- and `verify_paper_claims` bound one sentence per gap, so the restatements drifted
silently while the checked copy stayed right. The drift was real and measured: sigma moved from
five replicates to ten during the study, and an audit found the paper printing $3.16$ for $3.09$
twice, $1.01$ for $0.99$ three times (once in the SAME SENTENCE as a value that had been
updated), $2.15$ for $2.10$, $2.27$ for $2.22$ twice, and ``$5\\sigma$'' for $3.50\\sigma$ --- the
last off by a factor, because it divided by a sigma this study never estimated.

The property enforced is weaker than "every mention is bound to its own artifact" and much
stronger than nothing: **every sigma value the paper prints must appear as a scraped value
somewhere in the checker.** A restatement may share a checked sentence's number, which is the
point of a restatement; what it may not do is print a number no artifact produced. Every one of
the stale values above violates that, because each was a quotient of a superseded sigma.

Patterns are recovered from `verify_paper_claims.py` by parsing it, so a new sigma check is picked
up with no bookkeeping here. That direction matters: a coverage test nobody updates rots into a
tautology, which is precisely the failure this file exists to catch.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

CHECKER = pathlib.Path("scripts/verify_paper_claims.py")
PAPER = pathlib.Path("docs/paper/main.tex")

#: Any pattern that captures a number. Deliberately not restricted to sigma-bearing patterns: the
#: p0 swing is quoted as "2.5" in five places while the checker binds it through a displayed
#: equation whose pattern never mentions sigma, and a narrower scan would call that an orphan.
_CAPTURING = "([\\d.]+)"

#: value -> why the paper may print it in sigma units with no artifact behind it. Both entries are
#: pre-registered THRESHOLDS: numbers the study chose in advance to be judged against, not
#: measurements of anything. Keep this list at thresholds only. A measured quotient that cannot be
#: derived is not an exemption, it is the bug.
_EXEMPT: dict[float, str] = {
    2.0: (
        "The registered verdict rule for D4 (sec:budgetprereg) requires delta at the deepest cell "
        "to clear both +0.05 and 2 sigma. The 2 is the rule, fixed before the runs; sec:threats "
        "quotes it back when reporting that no cell reached it."
    ),
    1.0: (
        "sec:superseded partitions four non-zero superseded deltas by whether they sit under one "
        "sigma. The 1 is the partition, not a measurement; the four quotients it sorts ARE checked."
    ),
}


def _bound_values() -> set[float]:
    """Every number the checker's own regexes pull out of the paper."""
    src = CHECKER.read_text(encoding="utf-8")
    tex = PAPER.read_text(encoding="utf-8")
    out: set[float] = set()
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if _CAPTURING not in node.value:
            continue
        try:
            rx = re.compile(node.value, re.DOTALL | re.MULTILINE)
        except re.error:
            continue
        for m in rx.finditer(tex):
            for g in m.groups():
                try:
                    out.add(round(float(g), 4))
                except (TypeError, ValueError):
                    continue
    return out


def _sigma_mentions() -> list[tuple[int, float]]:
    """(line number, value) for every `$X\\sigma$` in the paper."""
    tex = PAPER.read_text(encoding="utf-8")
    return [
        (tex[: m.start()].count("\n") + 1, round(float(m.group(1)), 4))
        for m in re.finditer(r"([\d.]+)\\sigma", tex)
    ]


def test_every_sigma_value_is_derived_somewhere():
    bound = _bound_values()
    orphans: dict[float, list[int]] = {}
    for line, value in _sigma_mentions():
        if value not in bound and value not in _EXEMPT:
            orphans.setdefault(value, []).append(line)
    assert not orphans, (
        "these values are printed in sigma units in main.tex and no check in "
        f"verify_paper_claims.py derives them from an artifact: {orphans}. A sigma quotient that "
        "nothing derives is how this paper printed 3.16 for 3.09 and 5 for 3.50 --- add a check, "
        "or add the value to _EXEMPT with a reason it is a threshold rather than a measurement."
    )


def test_the_exemption_list_does_not_rot():
    printed = {v for _, v in _sigma_mentions()}
    stale = sorted(v for v in _EXEMPT if v not in printed)
    assert not stale, (
        f"_EXEMPT excuses {stale}, which the paper no longer prints in sigma units. Remove the "
        "entries so the list keeps meaning what it says."
    )


def test_coverage_is_not_vacuous():
    """A refactor that made either side return nothing would turn the test above into a tautology.

    Both floors are set well under the measured values (46 mentions, 14 distinct sigma quotients
    bound at the time of writing) so ordinary editing does not trip them, while an extraction that
    silently stopped finding patterns does.
    """
    mentions = _sigma_mentions()
    assert len(mentions) >= 30, f"expected the paper's sigma restatements, found {len(mentions)}"
    assert len({v for _, v in mentions}) >= 10, "sigma values look collapsed"
    assert len(_bound_values()) >= 100, "the checker's scraped-value set looks empty"


def test_a_stale_sigma_quotient_is_caught():
    """The guard has to FAIL on the bug it was built for, not merely pass on a clean paper.

    Replays the exact defect: 3.16 was a correct quotient of a five-replicate sigma that ten
    replicates superseded, so it survived every check while its own sentence's neighbour was
    fixed. Asserting on a mutated copy keeps the test honest about what it detects.
    """
    bound = _bound_values()
    assert 3.09 in bound, "the corrected quotient should be derived by a check"
    assert 3.16 not in bound, (
        "3.16 is derivable from some artifact, so this test can no longer distinguish the stale "
        "value from the correct one --- rewrite it against a value that is genuinely orphaned"
    )
