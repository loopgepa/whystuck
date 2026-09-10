r"""The planning document's own numbers, derived rather than typed.

`SUBMISSION.md` decides the submission order from how hard each paper leans on its companions,
and it carried those counts as prose. Nothing read them. Six review passes moved the papers and
the sentence stood still: it said Paper I cites Paper II nineteen times and Paper III twenty-one
while the sources measured 22 and 36.

Every number a paper prints is re-derived by `scripts/verify_paper_claims.py`. This file is the
one document that plans the submission and was exempt from that.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import paper_sources  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/papers/SUBMISSION.md"

#: Each paper, as the others refer to it: the `Paper~N` macro form and the descriptive form.
_REFERS_TO = {
    "p1-conditional-lever": ("Paper~I", "conditional lever"),
    "p2-repair-not-detection": ("Paper~II", "ceiling paper"),
    "p3-silent-no-op": ("Paper~III", "silent no-op"),
}


def _references(source: str, target: str) -> int:
    """How many times `source` refers to `target`, both forms counted.

    `Paper~I` needs the negative lookahead or it matches the first two characters of `Paper~II`
    and `Paper~III`, which would inflate every count into the paper with the shortest numeral.
    """
    tex = (ROOT / "docs/papers" / source / "main.tex").read_text(encoding="utf-8")
    macro, phrase = _REFERS_TO[target]
    return len(re.findall(re.escape(macro) + r"(?![I])", tex)) + len(
        re.findall(re.escape(phrase), tex, re.I)
    )


def test_the_dependency_table_matches_what_the_sources_measure() -> None:
    rows = []
    for source in paper_sources.PAPERS:
        for target in paper_sources.PAPERS:
            if source == target:
                continue
            rows.append((source, target, _references(source, target)))

    doc = DOC.read_text(encoding="utf-8")
    for source, target, n in rows:
        assert n > 0, f"{source} no longer refers to {target} at all; the table is meaningless"
        assert str(n) in doc, (
            f"{source} refers to {target} {n} times and SUBMISSION.md does not print {n}. "
            "The document is the stale one unless the papers changed on purpose."
        )


def test_paper_one_is_still_the_most_dependent_which_is_why_it_ships_first() -> None:
    """The submission order rests on this asymmetry, so the order is wrong if it inverts."""
    outbound = {
        s: sum(_references(s, t) for t in paper_sources.PAPERS if t != s)
        for s in paper_sources.PAPERS
    }
    most = max(outbound, key=lambda k: outbound[k])
    assert most == "p1-conditional-lever", (
        f"Paper I is no longer the most dependent ({outbound}); SUBMISSION.md orders the "
        "submission on the assumption that it is"
    )


def test_the_advertised_tmlr_page_counts_are_the_ones_the_build_produces() -> None:
    """37/24/20 are printed in the document; a restyle that moved them must move the text."""
    doc = DOC.read_text(encoding="utf-8")
    assert re.search(r"\|\s*I\s*\|\s*44\s*\|\s*\*\*37\*\*", doc), "Paper I's TMLR row is missing"
    assert re.search(r"\|\s*II\s*\|\s*28\s*\|\s*\*\*24\*\*", doc), "Paper II's TMLR row is missing"
    assert re.search(r"\|\s*III\s*\|\s*24\s*\|\s*\*\*20\*\*", doc), "Paper III's TMLR row missing"
