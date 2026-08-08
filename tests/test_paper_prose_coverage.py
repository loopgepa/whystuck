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

IT DID NOT ENFORCE THAT PROPERTY UNTIL 2026-08-05. For its whole life `_bound_values` also
replayed every string constant in the checker containing the capture fragment `([\\d.]+)` against
the document, on the theory that a scrape pattern locates the site it pins; the checker composes
its table rows out of a local `num = r"([\\d.]+)"`, and that constant compiled on its own matched
3937 numbers, so the bound set was the document and the assertion was that the paper agrees with
itself. It was green throughout and concealed seven orphans, four of them in the very species the
paragraphs above were written about --- including `$+0.0017$`, which sat as a literal inside the
pattern checking its own sigma quotient, the third bullet's defect recurring verbatim.

Two lessons are now encoded here rather than described. First, `test_coverage_is_not_vacuous`
could not have caught this: an over-broad bound set is LARGER than a correct one, so every size
floor passes. Only `test_the_guard_reports_an_injected_orphan`, which feeds in a decimal known to
be underivable and demands a failure, distinguishes a guard that binds from one that merely
returns. Second, the replay was a second implementation of something the check list already
records exactly --- see `_bound_values` --- and it is the reimplementation, not the original, that
went wrong.
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
    """Every paper-side number some check pins to an artifact.

    A check is `(label, paper_value, tolerance, artifact_value)` and `main()` asserts
    `|artifact - paper| <= tol`, so column 1 IS the record of which printed numbers are pinned:
    for a scraped check it is the value read out of the `.tex`, and for a derived one it is the
    quantity the paper is asserted to print. Reading the pinning out of the check list is the only
    honest way to ask the question, because the check list is what actually runs.

    An earlier version also replayed every string constant in the checker that contained the
    capture fragment `([\\d.]+)` against the whole document, on the theory that a scrape pattern
    locates the site it pins. The flaw is that a pattern under construction is indistinguishable
    from a finished one once you have only the string: the checker builds table rows out of a local
    `num = r"([\\d.]+)"`, and that constant compiled alone matched 3937 numbers, binding the paper
    to itself. Measured at the repair: the replay's ONLY contribution beyond the check list was the
    seven orphans it concealed, so nothing legitimate was lost by deleting it.
    """
    return {round(float(c[1]), 4) for c in vpc._checks() if isinstance(c[1], (int, float))}


def _prose() -> tuple[str, list[str]]:
    """The paper with tabular bodies blanked, newlines preserved so line numbers survive."""
    tex = PAPERS_TEX()
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


def test_the_guard_reports_an_injected_orphan(monkeypatch):
    """The guard must FAIL on a decimal no check derives. Nothing else here proves that.

    This is the test the guard needed and did not have. `_bound_values` used to replay every
    string constant in the checker that contained the capture fragment `([\\d.]+)` against the whole
    document, and one of those constants IS that fragment, bare --- a local named `num`, composed
    into anchored row patterns a few lines later. Compiled alone it matched 3937 numbers, so the
    bound set absorbed the paper wholesale and `test_every_prose_decimal_is_derived` asserted that
    the paper agrees with itself. It passed for its whole life and was hiding seven orphans.

    The size floors below cannot catch that: a bound set inflated to swallow everything is LARGER
    than a correct one, so it clears every floor. Only a value known to be underivable can
    distinguish a guard that binds from a guard that merely returns.
    """
    injected = 3.1416
    assert injected not in _bound_values(), "pick a value the checker really cannot derive"
    monkeypatch.setitem(
        globals(),
        "PAPERS_TEX",
        lambda _real=PAPERS_TEX: _real() + f"\n\\noindent An orphan: ${injected}$.\n",
    )
    try:
        test_every_prose_decimal_is_derived()
    except AssertionError as exc:
        assert str(injected) in str(exc), f"guard fired but not about {injected}: {exc}"
    else:
        raise AssertionError(
            f"the guard passed a prose decimal ({injected}) that no check derives, so it is "
            "vacuous: something in _bound_values is matching the paper against itself."
        )


def test_the_exemption_list_does_not_rot():
    printed = {abs(v) for _, v in _mentions()}
    stale = sorted(v for v in _EXEMPT if v not in printed)
    assert not stale, f"_EXEMPT excuses {stale}, which the paper no longer prints. Remove them."


def test_coverage_is_not_vacuous():
    """Either side COLLAPSING would turn the test above into a tautology.

    Floors are set from the MEASURED sizes at the time of writing --- 339 prose decimals, 127 of
    them distinct, against a bound set of 318 --- at roughly three quarters of each, so ordinary
    editing does not trip them while a silently broken extraction does. They were not guessed: a
    first draft of this test asserted 250 and failed on a correct paper, which is the same
    unfalsifiable-threshold mistake the guards exist to catch, one level up.

    Read the name narrowly. These floors catch a bound set that has gone EMPTY, and that is all
    they ever caught. The vacuity that actually happened ran the other way: the bound set swelled
    to include the whole document, which clears every floor here by a wide margin and made the
    guard unfalsifiable while these three assertions passed. `test_the_guard_reports_an_injected_
    orphan` is the one that detects that direction, and it is the reason these numbers moved.
    """
    mentions = _mentions()
    assert len(mentions) >= 250, f"expected the paper's prose decimals, found {len(mentions)}"
    assert len({v for _, v in mentions}) >= 95, "prose values look collapsed"
    assert len(_bound_values()) >= 235, "the checker's derived-value set looks empty"


def test_tables_are_excluded_so_the_table_guard_stays_responsible():
    """The blanking must actually remove tabular bodies, or the two guards silently overlap and a
    regression in either looks covered by the other."""
    prose, _ = _prose()
    tex = PAPERS_TEX()
    assert r"\begin{tabular}" in tex, "no tables found; the fixture is wrong, not the paper"
    assert len(_NUM.findall(prose)) < len(_NUM.findall(tex)), "tabular bodies were not blanked"
    assert len(prose.split("\n")) == len(tex.split("\n")), "blanking changed the line count"
