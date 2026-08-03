"""The papers' own navigation documents must agree with each other and with the sources.

Added 2026-08-03 after finding two stale numbers that every other guard was blind to. The 788
claim checks scrape the papers' `.tex`; nothing looked at the Markdown around them. So
`docs/papers/README.md` still gave Paper I as 33 pages and Paper II as 17 after both grew by one,
and `docs/papers/SUBMISSION.md` still described the pre-split monolith -- 45 pages, 16 tables, 28
references -- for a document that no longer exists.

That is a restated-statistic defect (the same quantity given two values in two places), which is the
class the claim checks were built for, in files the claim checks do not read. The fix is not to
remember harder.

Page counts are compared between the two documents rather than against a built PDF, because the PDFs
are build artifacts and are not in a clean checkout. Table, figure and citation counts *are* derived
from source, so those are checked against it.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPERS = Path(__file__).resolve().parent.parent / "docs/papers"
SLUGS = ("p1-conditional-lever", "p2-repair-not-detection", "p3-silent-no-op")


def _readme_pages() -> dict[str, int]:
    """Slug -> page count, from the overview table's last column."""
    out = {}
    for line in (PAPERS / "README.md").read_text(encoding="utf-8").splitlines():
        for slug in SLUGS:
            if f"`{slug}/`" in line and line.startswith("|"):
                cells = [c.strip() for c in line.split("|")]
                out[slug] = int(cells[-2])
    return out


def _submission_table() -> dict[str, dict[str, int]]:
    """Slug -> {pages, tables, figures, citations}, from the Size table."""
    out = {}
    for line in (PAPERS / "SUBMISSION.md").read_text(encoding="utf-8").splitlines():
        for slug in SLUGS:
            if f"`{slug}`" in line and line.startswith("|"):
                cells = [c.strip() for c in line.split("|")]
                nums = [int(c) for c in cells if re.fullmatch(r"\d+", c)]
                out[slug] = dict(
                    zip(("pages", "tables", "figures", "citations"), nums, strict=True)
                )
    return out


def _source_counts(slug: str) -> dict[str, int]:
    tex = (PAPERS / slug / "main.tex").read_text(encoding="utf-8")
    cites: set[str] = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex):
        cites.update(k.strip() for k in group.split(",") if k.strip())
    return {
        "tables": len(re.findall(r"\\begin\{table", tex)),
        "figures": len(re.findall(r"\\includegraphics", tex)),
        "citations": len(cites),
    }


def test_every_paper_appears_in_both_documents() -> None:
    assert set(_readme_pages()) == set(SLUGS)
    assert set(_submission_table()) == set(SLUGS)


def test_page_counts_agree_between_readme_and_submission() -> None:
    readme, submission = _readme_pages(), _submission_table()
    for slug in SLUGS:
        assert readme[slug] == submission[slug]["pages"], (
            f"{slug}: README says {readme[slug]} pages, SUBMISSION says {submission[slug]['pages']}"
        )


def test_submission_table_matches_the_sources_it_describes() -> None:
    submission = _submission_table()
    for slug in SLUGS:
        src = _source_counts(slug)
        for key, n in src.items():
            assert submission[slug][key] == n, (
                f"{slug}: SUBMISSION says {submission[slug][key]} {key}, source has {n}"
            )


def test_submission_no_longer_states_monolith_size_as_current() -> None:
    """The stale figures may be *mentioned* as withdrawn, which is the corrections practice, but
    not asserted. Guard the specific sentence shape rather than the numerals."""
    text = (PAPERS / "SUBMISSION.md").read_text(encoding="utf-8")
    assert "45 pages: 38 of main text" not in text
    assert "pre-split monolith" in text, "the withdrawal should stay on the record"
