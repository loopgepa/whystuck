"""Every decimal the paper states in prose must be one an artifact produces.

WHY THIS TEST EXISTS. It is the third and widest of the coverage guards.
`test_paper_table_coverage` covers numbers printed inside tables and
`test_paper_sigma_coverage` covers quotients printed in units of sigma; both were written after a
number went stale in a place nothing was looking. This one covers the rest of the prose, and it
was written after an audit of that remaining surface found four unbound decimals, three of which
turned out to matter:

* the per-arm drops from baseline, $d_B$ and $d_D$, printed beside the contrast they compose. Two
  of their six values happened to collide with checked numbers and one did not. They are a
  DIFFERENT quantity from $\\dd = d_D - d_B$, which is precisely the confusion the paper's own
  decomposition section exists to prevent, so they are now checked as vectors and their difference
  is checked against the published contrast.
* the headroom at the saturated $p_0$ level, which is the number the paper's admission of a
  falsified prediction rests on.
* the superseded first attempt's $\\dd$. Its sigma quotient was checked while the delta itself sat
  as a literal inside the checking pattern, so the quotient was pinned to a value nothing verified.
  Its artifact is `validation-study.json`, the one released run whose name carries no suffix --- a
  glob over `validation-study-*.json` misses it, which is how it stayed unbound.

The property enforced is the same as the sigma guard's: a value may be restated, but every value
the paper prints must be one some check derives from an artifact. Only genuine constants are
exempt, and there is exactly one.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import verify_paper_claims as vpc  # noqa: E402

CHECKER = pathlib.Path("scripts/verify_paper_claims.py")
PAPER = pathlib.Path("docs/paper/main.tex")

#: Prose decimals, i.e. inside inline math. Two or more decimal places, so section numbers and
#: LaTeX lengths do not enter. Tables are excluded because the table guard owns them.
_NUM = re.compile(r"\$([+-]?)(\d+\.\d{2,4})\$")

#: value -> why no artifact can produce it. Constants of statistics only. A measured quantity that
#: cannot be derived is not an exemption, it is the finding.
_EXEMPT: dict[float, str] = {
    1.96: (
        "The normal-approximation critical value, quoted in sec:repeat to say the study does NOT "
        "use it: at K=5 the correct values are t(4) = 2.132 and 2.776, and a normal approximation "
        "would call one of the three levels significant when t does not. Both t values ARE checked."
    ),
}


def _bound_values() -> set[float]:
    """Every number the checker derives, plus every number its regexes pull from the paper."""
    src = CHECKER.read_text(encoding="utf-8")
    tex = PAPER.read_text(encoding="utf-8")
    out = {round(float(c[1]), 4) for c in vpc._checks() if isinstance(c[1], (int, float))}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if "([\\d.]+)" not in node.value:
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


def _prose() -> tuple[str, list[str]]:
    """The paper with tabular bodies blanked, newlines preserved so line numbers survive."""
    tex = PAPER.read_text(encoding="utf-8")
    prose = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        lambda m: re.sub(r"[^\n]", " ", m.group(0)),
        tex,
        flags=re.DOTALL,
    )
    return prose, prose.split("\n")


def _mentions() -> list[tuple[int, float]]:
    prose, _ = _prose()
    return [
        (prose[: m.start()].count("\n") + 1, round(float(m.group(1) + m.group(2)), 4))
        for m in _NUM.finditer(prose)
    ]


def test_every_prose_decimal_is_derived():
    bound = _bound_values()
    _, lines = _prose()
    orphans: dict[float, list[int]] = {}
    for line, value in _mentions():
        # Sign is not load-bearing for provenance: the paper writes a magnitude where the artifact
        # carries a signed value and vice versa (a "loss of $0.042$" against a delta of $-0.042$).
        if value in bound or -value in bound or abs(value) in _EXEMPT:
            continue
        orphans.setdefault(value, []).append(line)
    detail = {
        v: [f"L{ln}: {lines[ln - 1].strip()[:90]}" for ln in lns] for v, lns in orphans.items()
    }
    assert not orphans, (
        "these decimals are stated in main.tex prose and no check derives them from an artifact: "
        f"{detail}. Add a check to verify_paper_claims._checks(), or add the value to _EXEMPT with "
        "a reason it is a constant rather than a measurement."
    )


def test_the_exemption_list_does_not_rot():
    printed = {abs(v) for _, v in _mentions()}
    stale = sorted(v for v in _EXEMPT if v not in printed)
    assert not stale, f"_EXEMPT excuses {stale}, which the paper no longer prints. Remove them."


def test_coverage_is_not_vacuous():
    """Either side collapsing would turn the test above into a tautology.

    Floors are set from the MEASURED sizes at the time of writing --- 237 prose decimals, 101 of
    them distinct, against a bound set of 270 --- at roughly three quarters of each, so ordinary
    editing does not trip them while a silently broken extraction does. They were not guessed: a
    first draft of this test asserted 250 and failed on a correct paper, which is the same
    unfalsifiable-threshold mistake the guards exist to catch, one level up.
    """
    mentions = _mentions()
    assert len(mentions) >= 180, f"expected the paper's prose decimals, found {len(mentions)}"
    assert len({v for _, v in mentions}) >= 75, "prose values look collapsed"
    assert len(_bound_values()) >= 200, "the checker's derived-value set looks empty"


def test_tables_are_excluded_so_the_table_guard_stays_responsible():
    """The blanking must actually remove tabular bodies, or the two guards silently overlap and a
    regression in either looks covered by the other."""
    prose, _ = _prose()
    tex = PAPER.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex, "no tables found; the fixture is wrong, not the paper"
    assert len(_NUM.findall(prose)) < len(_NUM.findall(tex)), "tabular bodies were not blanked"
    assert len(prose.split("\n")) == len(tex.split("\n")), "blanking changed the line count"
