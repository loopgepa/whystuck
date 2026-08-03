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

A third instance turned up in the same sweep, and it is the worst of the three because it is the
number a reader checks first: three documents advertised the verifier as "762 checks" while the
script prints `788/788 paper claims match the released artifacts` on every run. The stated count is
now compared against `len(_checks())` directly, so the advertisement and the thing advertised cannot
part company again.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from verify_paper_claims import _checks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "docs/papers"
SLUGS = ("p1-conditional-lever", "p2-repair-not-detection", "p3-silent-no-op")

#: Released documents that advertise how many checks the verifier runs. Not every doc has to, but
#: any that does has to be right, and a new one is caught by the "somebody says it" assertion below.
DOCS_STATING_CHECK_COUNT = ("docs/EXPERIMENTS.md", "docs/papers/README.md")
CHECK_COUNT_RE = re.compile(r"(\d{2,5})\s+(?:machine-)?checks?\b")

#: The README advertises the suite's size. That number goes stale on the commit that adds a test,
#: which is how it was already wrong by four: the commit adding this very file broke it.
TEST_COUNT_RE = re.compile(r"\*\*([\d,]+) tests\*\*")
#: Set in the child so a collection sweep can never re-enter the guard that spawned it.
NO_REENTER = "WHYSTUCK_SKIP_COLLECT_GUARD"


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


def test_every_advertised_check_count_equals_the_verifier_s_own() -> None:
    """The number in the docs is the number the script prints, or it is a lie with a shelf life."""
    actual = len(_checks())
    stated: list[tuple[str, int, str]] = []
    for rel in DOCS_STATING_CHECK_COUNT:
        for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            for m in CHECK_COUNT_RE.finditer(line):
                stated.append((f"{rel}:{i}", int(m.group(1)), line.strip()))
    assert stated, "no document states the check count; this guard would pass vacuously"
    wrong = [(where, n, line) for where, n, line in stated if n != actual]
    assert not wrong, "\n".join(
        f"{where} says {n} checks, verify_paper_claims runs {actual}: {line[:90]}"
        for where, n, line in wrong
    )


def test_readme_test_count_equals_what_pytest_collects() -> None:
    """Costs one collection sweep (~20 s) and belongs to the pre-push gate, not pre-commit.

    Worth that: the number is the repository's main credibility signal, it moves on any commit that
    adds a test, and it had already drifted by four before anyone noticed. When this fails, the
    message carries the new number -- put it in the README rather than deleting the assertion.
    """
    if os.environ.get(NO_REENTER):
        pytest.skip("inside a collection sweep spawned by this guard")
    stated = TEST_COUNT_RE.search((ROOT / "README.md").read_text(encoding="utf-8"))
    assert stated, "README no longer advertises a test count; drop this guard or restore the number"
    claimed = int(stated.group(1).replace(",", ""))

    env = {**os.environ, NO_REENTER: "1"}
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    m = re.search(r"(\d+) tests collected", out.stdout)
    assert m, f"could not read a collection count from pytest:\n{out.stdout[-800:]}"
    collected = int(m.group(1))
    assert claimed == collected, (
        f"README advertises {claimed} tests, pytest collects {collected}. "
        f"The README is the stale one unless a test was deleted on purpose."
    )


def test_submission_no_longer_states_monolith_size_as_current() -> None:
    """The stale figures may be *mentioned* as withdrawn, which is the corrections practice, but
    not asserted. Guard the specific sentence shape rather than the numerals."""
    text = (PAPERS / "SUBMISSION.md").read_text(encoding="utf-8")
    assert "45 pages: 38 of main text" not in text
    assert "pre-split monolith" in text, "the withdrawal should stay on the record"
