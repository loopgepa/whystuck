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

import datetime as _dt
import itertools
import json
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

#: SUBMISSION.md's abstract-length sentence: "Abstract lengths against the 1920-character metadata
#: field: **1912**, **1895**, 1786." The limit and all three lengths, in paper order. Bold marks
#: the ones close enough to the limit that the document calls out their headroom, so which entries
#: carry it changes as abstracts grow; the guard reads the numbers and ignores the emphasis.
ABSTRACT_LENGTHS_RE = re.compile(
    r"Abstract lengths against the (\d+)-character metadata\s+field: "
    r"\*{0,2}(\d+)\*{0,2}, \*{0,2}(\d+)\*{0,2}, \*{0,2}(\d+)\*{0,2}"
)

#: The same sentence's three headroom figures. Restated arithmetic over two numbers already in
#: the sentence, which is the class this file exists for: the lengths were guarded and the
#: subtraction beside them was not, so a one-character edit to an abstract left the headroom
#: reading 15 where it was 14 until someone did the sum by hand.
ABSTRACT_HEADROOM_RE = re.compile(r"The headroom is (\d+), (\d+) and (\d+) characters")


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
    """Tables, figures and distinct citations in one paper's source, comments excluded.

    The exclusion is the load-bearing line. Every `main.tex` opens with a comment explaining why
    `\\pdfoutput=1` has to sit in the first five lines, and that comment contains the word
    `\\includegraphics`. Counting it put every paper's figure count one too high, so SUBMISSION.md
    advertised three figures for Paper I, which has two, and one for Paper III, which had none. The
    guard agreed with the document because both counted the same comment: the shared-error blind
    spot these papers are about, inside the test that exists to stop it.
    """
    # The EXPANSION, not main.tex. bibtex reads what the paper inputs, so a citation living in
    # `shared/loop.tex` is in that paper's reference list; counting main.tex alone undercounted
    # every paper -- Paper III by two of its sixteen -- and the document agreed because it was
    # derived from the same short read. `_expanded` is defined below and resolved at call time.
    tex = "\n".join(
        line for line in _expanded(slug).splitlines() if not line.lstrip().startswith("%")
    )
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


def test_advertised_abstract_lengths_equal_the_measured_ones() -> None:
    """SUBMISSION.md's three abstract lengths, and the limit they are measured against.

    Written after finding Paper III's advertised at 1786 while the source measured 1788: an edit had
    shortened the abstract by two characters and the document kept the old figure. Nothing looked at
    it, because the numbers in SUBMISSION.md's *table* were guarded and this sentence's were not.

    Paper I sits eight characters under the limit and the document says so, which is what makes
    this worth a test rather than a note: a stale length there is the difference between a
    submission that goes through and one arXiv's form rejects, and the drift direction that
    matters -- an abstract growing while the number stays put -- is the one already observed.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_arxiv_package as pkg

    text = (PAPERS / "SUBMISSION.md").read_text(encoding="utf-8")
    m = ABSTRACT_LENGTHS_RE.search(text)
    assert m, (
        "SUBMISSION.md no longer states abstract lengths in the form this guard reads. Restore the "
        "sentence or update ABSTRACT_LENGTHS_RE -- do not delete the guard."
    )
    assert int(m.group(1)) == pkg.ABSTRACT_LIMIT, (
        f"SUBMISSION.md cites a {m.group(1)}-character limit, make_arxiv_package uses "
        f"{pkg.ABSTRACT_LIMIT}"
    )
    for slug, claimed in zip(SLUGS, m.groups()[1:], strict=True):
        measured = len(pkg.plain_abstract(PAPERS / slug))
        assert int(claimed) == measured, (
            f"{slug}: SUBMISSION.md advertises a {claimed}-character abstract, the source measures "
            f"{measured}. The document is the stale one unless the abstract was just edited."
        )

    head = ABSTRACT_HEADROOM_RE.search(text)
    assert head, (
        "SUBMISSION.md no longer states the three headroom figures in the form this guard reads"
    )
    for slug, claimed in zip(SLUGS, head.groups(), strict=True):
        left = pkg.ABSTRACT_LIMIT - len(pkg.plain_abstract(PAPERS / slug))
        assert int(claimed) == left, (
            f"{slug}: SUBMISSION.md gives {claimed} characters of headroom, the abstract leaves "
            f"{left}. The headroom is the limit minus the length and both are stated above it."
        )


def test_the_conference_limit_multiples_are_the_stated_lengths_divided_by_the_limit() -> None:
    """SUBMISSION.md's "Paper I runs 3.0x" against a 9-page main-text limit.

    It read "about four times" from the split until 2026-08-17, which is 39/9 rather than 27/9: the
    sentence divided the total page count by a limit that governs main text only. Nothing checked
    it, because it is arithmetic over two numbers stated in the same document and neither the page
    guard in `make_arxiv_package` nor the table guards above look at prose multiples.

    Derived rather than pinned: the multiples come from the main-text lengths the file states two
    paragraphs earlier and the limit it names in the same sentence, so an edit to any of the three
    has to move the others.
    """
    text = (PAPERS / "SUBMISSION.md").read_text(encoding="utf-8")
    lengths = re.search(r"\*\*Main-text length is (\d+) / (\d+) / (\d+) pages\*\*", text)
    assert lengths, "SUBMISSION.md no longer states the three main-text lengths in a readable form"
    limit = re.search(r"the (\d+)-page\s+main-text limit", text)
    assert limit, "SUBMISSION.md no longer names the conference main-text limit"
    numeral = {"I": 0, "II": 1, "III": 2}
    multiples = re.findall(r"Paper (I{1,3}) (?:runs )?([\d.]+)x", text)
    assert len(multiples) == len(SLUGS), (
        f"found {len(multiples)} stated multiples, expected {len(SLUGS)}; the sentence has been "
        "reworded out of this guard's reach"
    )
    for roman, claimed in multiples:
        main = int(lengths.group(numeral[roman] + 1))
        expected = round(main / int(limit.group(1)), 1)
        assert float(claimed) == expected, (
            f"Paper {roman} is advertised at {claimed}x the {limit.group(1)}-page limit; "
            f"{main} main-text pages over that limit is {expected}x"
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


#: docs/papers/README.md's one-line summary of the artifact release: "34 study runs, 225 replicate
#: scorings, 258 cassettes."
RELEASE_SIZE_RE = re.compile(r"(\d+) study runs, (\d+) replicate scorings, (\d+) cassettes")


def test_advertised_release_size_equals_what_is_committed() -> None:
    """The three release counts, against the files actually in the tree.

    Found stale by exactly one on each of two of the three: the cross-provider run added a study
    JSON and a cassette, and the sentence kept saying 33 and 257. That is the drift direction a
    reader cannot detect -- an artifact release that quietly grows while its advertisement stands
    still -- and it is the same restated-statistic class as the page and check counts above, so it
    gets the same treatment rather than a note.

    Runs are counted from `docs/`, cassettes from the fixture tree, so the two sides of the
    sentence come from different places and cannot be made to agree by editing one of them.
    """
    text = (PAPERS / "README.md").read_text(encoding="utf-8")
    m = RELEASE_SIZE_RE.search(text)
    assert m, (
        "docs/papers/README.md no longer states the release size in the form this guard reads. "
        "Restore the sentence or update RELEASE_SIZE_RE -- do not delete the guard."
    )
    runs, replicates, cassettes = (int(g) for g in m.groups())
    cassette_dir = ROOT / "tests/fixtures/cassettes"

    assert runs == len(list((ROOT / "docs").glob("validation-study*.json")))
    assert replicates == len(list((cassette_dir / "repeat").glob("*.json")))
    assert cassettes == len(list(cassette_dir.rglob("*.json")))


#: The same two release counts, restated in `docs/EXPERIMENTS.md`'s artifact table. The guard
#: above covered `docs/papers/README.md` alone, and the sibling document that states the SAME
#: quantities was found at 33 runs and 257 cassettes against a tree holding 43 and 267 -- stale by
#: exactly ten on both. Guarding one of two documents that restate a statistic is not guarding the
#: statistic; it just moves the drift somewhere nothing looks.
EXPERIMENTS_RUNS_RE = re.compile(r"\| `docs/validation-study-\*\.json` \| (\d+) study runs")
EXPERIMENTS_CASSETTES_RE = re.compile(r"\| `tests/fixtures/cassettes/\*\*` \| (\d+) cassettes")


def test_experiments_md_release_counts_equal_what_is_committed() -> None:
    text = (ROOT / "docs/EXPERIMENTS.md").read_text(encoding="utf-8")
    cassette_dir = ROOT / "tests/fixtures/cassettes"
    for label, pattern, actual in (
        (
            "study runs",
            EXPERIMENTS_RUNS_RE,
            len(list((ROOT / "docs").glob("validation-study*.json"))),
        ),
        ("cassettes", EXPERIMENTS_CASSETTES_RE, len(list(cassette_dir.rglob("*.json")))),
    ):
        m = pattern.search(text)
        assert m, (
            f"docs/EXPERIMENTS.md no longer states its {label} count in the form this guard "
            "reads. Restore the row or update the pattern -- do not delete the guard."
        )
        assert int(m.group(1)) == actual, (
            f"docs/EXPERIMENTS.md advertises {m.group(1)} {label}, the tree holds {actual}"
        )


#: The release's worker census, as `docs/papers/README.md` states it. Worth its own guard because
#: the sentence it replaced said "34 of the runs are the papers' own" -- a category, not a
#: measurement, and wrong by one the moment you ask which worker: 33 are on `gpt-4.1-mini` and the
#: 34th is the `gpt-4.1-nano` worker swap. Aggregating a category with an exception in it is how
#: every restated statistic in this register went stale, so the census is now four counts against
#: four model ids read out of the run metadata.
WORKER_CENSUS_RE = re.compile(
    r"\*\*(\d+)\*\* runs are on\s+`(\S+)`.*?"
    r"\*\*(\d+)\*\* is the\s+`(\S+)` worker swap.*?"
    r"\*\*(\d+)\*\* are on\s+`(\S+)`.*?"
    r"and \*\*(\d+)\*\* on `(\S+)`",
    re.DOTALL,
)


def test_advertised_worker_census_equals_the_run_metadata() -> None:
    import collections

    census = collections.Counter(
        json.loads(p.read_text(encoding="utf-8"))["meta"]["worker_model"]
        for p in (ROOT / "docs").glob("validation-study*.json")
    )
    m = WORKER_CENSUS_RE.search((PAPERS / "README.md").read_text(encoding="utf-8"))
    assert m, (
        "docs/papers/README.md no longer states the worker census in the form this guard reads. "
        "Restore the sentence or update WORKER_CENSUS_RE -- do not delete the guard."
    )
    claimed = {m.group(i + 1): int(m.group(i)) for i in (1, 3, 5, 7)}
    assert claimed == dict(census), (
        f"README advertises the worker census as {claimed}, the run metadata says {dict(census)}"
    )
    assert sum(census.values()) == len(list((ROOT / "docs").glob("validation-study*.json")))


def test_the_paper_licence_grants_over_paths_that_exist() -> None:
    """The CC BY grant enumerates what it covers, and a path migration invalidated that list once.

    It was authored beside a single `main.tex` and read "the PAPER in this directory: main.tex,
    references.bib, the figures under figures/, SUBMISSION.md, REPRODUCE.md". The three-way split
    moved it into `shared/`, where none of `main.tex`, `SUBMISSION.md` or `REPRODUCE.md` exists, so
    for the whole life of the split tree the grant named a layout that had stopped existing.
    Nothing caught it because a licence is prose and no guard reads prose. This one does.

    Paths resolve against `docs/papers/`, which is what the licence now says they do.
    """
    text = (PAPERS / "shared" / "LICENSE").read_text(encoding="utf-8")
    # The GRANT only, not the paragraph explaining it. That paragraph quotes the pre-split wording,
    # which names a bare `main.tex`, so a guard reading it would fail on the history it records.
    start = text.index("Paths are relative to docs/papers/")
    scope = text[start : text.index("and any PDF built from them.", start)]
    named = set()
    for raw in re.split(r"[\s,;:]+", scope):
        tok = raw.strip("().\"'")
        if tok.endswith("/") or re.search(r"\.(tex|md)$", tok):
            named.add(tok.removeprefix("docs/papers/"))
    assert len(named) >= 8, (
        f"the licence scope parsed to {len(named)} paths, so this guard would be nearly empty; "
        f"the sentence shape changed: {sorted(named)}"
    )
    missing = sorted(n for n in named if not (PAPERS / n).exists())
    assert not missing, (
        f"the paper licence grants over {missing}, which do not exist under docs/papers/. "
        "A licence naming files that are not there grants nothing over the files that are."
    )


def test_every_extra_the_docs_tell_a_reader_to_install_exists() -> None:
    """`pip install -e '.[X]'` fails open: pip warns and installs the base package anyway.

    `docs/papers/REPRODUCE.md` told readers to run `pip install -e '.[dev]'` for the whole life
    of this file. `dev` is a PEP 735 entry under `[dependency-groups]`, not an extra, so pip
    printed "does not provide the extra 'dev'" and installed nothing else -- leaving the reader
    with no ruff, no mypy and no pytest, and every command in the Gates section failing on a
    missing binary. Nothing caught it because the install still exited 0.

    So: every extra any document names has to be a real extra. Read from pyproject rather than
    listed here, or this guard would need editing on the commit that renames one.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras_block = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", pyproject, re.M | re.S
    )
    assert extras_block, "pyproject has no [project.optional-dependencies]; update this guard"
    extras = set(re.findall(r"^([A-Za-z][\w-]*)\s*=", extras_block.group(1), re.M))
    assert extras, "parsed no extras out of pyproject; the guard would be vacuous"

    named: dict[str, set[str]] = {}
    for rel in ("docs/papers/REPRODUCE.md", "docs/EXPERIMENTS.md", "README.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for extra in re.findall(r"-e ['\"]?\.\[([\w-]+)\]", text):
            named.setdefault(extra, set()).add(rel)
    assert named, "no document names an extra at all; the guard would be vacuous"

    bogus = {e: sorted(f) for e, f in named.items() if e not in extras}
    assert not bogus, (
        f"documents tell a reader to install extras that do not exist: {bogus}. "
        f"pyproject defines {sorted(extras)}. A PEP 735 dependency group is not an extra and "
        "pip will not install it this way."
    )


#: Roman numeral -> slug, for pointers of the form "entry~N of Paper~III's register".
_PAPER_BY_NUMERAL = {
    "I": "p1-conditional-lever",
    "II": "p2-repair-not-detection",
    "III": "p3-silent-no-op",
}


def _register_entries(slug: str) -> list[str]:
    """One paper's corrections register, split into entries in printed order."""
    text = (PAPERS / slug / "main.tex").read_text(encoding="utf-8")
    start = text.index("entries follow")
    end = text.index("\\end{enumerate}", start)
    # `\item` is a prefix of `\itemsep`, so a bare split over-counts every list by its spacing
    # directives. That mistake was made once already while auditing these very counts.
    return [e for e in re.split(r"\\item(?![A-Za-z])", text[start:end])[1:]]


def test_cross_paper_register_pointers_land_on_the_entry_they_describe() -> None:
    """A pointer into ANOTHER paper's register is positional, so inserting an entry breaks it.

    That is not hypothetical. Paper II cites the entry withdrawing "four runs overrode" as
    `entry~14 of Paper~III's register`. Two entries were later inserted above it during review,
    which moved it to 16 and left Paper II pointing at "Ten of the fifteen reported a delta of
    exactly zero" -- a real, published-facing mis-citation with nothing watching it, since a
    number that still resolves to *an* entry breaks no build and no other guard.

    So: resolve every such pointer and require the quoted phrase in the citing sentence to occur
    in the entry it names. Case-insensitive, because the citing sentence lowercases the quote.
    """
    checked = 0
    for slug in SLUGS:
        flat = re.sub(r"\s+", " ", (PAPERS / slug / "main.tex").read_text(encoding="utf-8"))
        for m in re.finditer(r"entry~(\d+) of Paper~(I{1,3})'s register", flat):
            n, numeral = int(m.group(1)), m.group(2)
            target = _PAPER_BY_NUMERAL[numeral]
            assert target != slug, f"{slug} cites its own register as {numeral}"
            entries = _register_entries(target)
            assert 1 <= n <= len(entries), (
                f"{slug} cites entry {n} of {target}'s register, which holds {len(entries)}"
            )
            quotes = re.findall(r"``(.+?)''", flat[max(0, m.start() - 240) : m.start()])
            assert quotes, (
                f"{slug}'s pointer at entry~{n} of {target} quotes nothing, so this guard cannot "
                "confirm it lands correctly; quote the claim or drop the number"
            )
            phrase = quotes[-1].lower()
            body = re.sub(r"\s+", " ", entries[n - 1]).lower()
            assert phrase in body, (
                f"{slug} cites entry~{n} of {target}'s register for {quotes[-1]!r}, but that "
                f"entry reads {body[:120]!r}. An entry was probably inserted above it."
            )
            checked += 1
    assert checked, (
        "no cross-paper register pointer found; this guard has stopped covering anything"
    )


#: Index -> a phrase that must still occur in that row of Paper III's defect table. The series
#: cites these defects positionally 21 times, across all three papers and the shared harness
#: section, and always by paraphrase ("the selector confound", "a ninth", "defect~1's mirror
#: image"), never by the table's own wording. So no textual overlap between a citation and its
#: row can be required: every one of those 21 sites would fail it. What CAN be pinned is the
#: numbering the citations were written against, which protects all of them at once.
_DEFECT_IDENTITY = {
    1: "Critique gene never changed",
    2: "Reflective dataset component-agnostic",
    3: "round\\_robin",
    4: "dropped the largest atom",
    5: "Seeds re-partition",
    6: "own ground-truth signal was a",
    7: "swallowed a cassette miss",
    8: "empty reflection completion",
    9: "arm accepted nothing",
    10: "reached only the",
    11: "pooled-$\\sigma$ estimator",
}


def _defect_rows() -> dict[int, str]:
    """Paper III's defect table, row number -> the defect cell, in printed order."""
    text = (PAPERS / "p3-silent-no-op/main.tex").read_text(encoding="utf-8")
    label = text.index("\\label{tab:defects}")
    table = text[text.rindex("\\begin{table}", 0, label) : label]
    # The .tex is hard-wrapped, so a row's two cells are split by " &\n" as often as by " & ".
    # Matching on the unflattened text drops whichever rows happen to wrap at the separator, which
    # silently cost this parse row 8 on the first attempt.
    flat = re.sub(r"\s+", " ", table)
    rows = {}
    for chunk in flat.split("\\\\"):
        m = re.match(r"\s*(?:\\midrule\s*)?(\d+) & (.+?) & ", chunk)
        if m:
            rows[int(m.group(1))] = m.group(2)
    return rows


def test_the_defect_numbering_the_series_cites_has_not_shifted() -> None:
    """Defect indices are positional pointers into another paper's list, like `entry~N` above.

    `verify_paper_claims` already derives the table's total and asserts the rows run 1..n
    contiguously, and the guard above protects register pointers. Neither covers this: inserting a
    twelfth defect in the middle keeps the numbering contiguous, moves the derived counts with it,
    and silently repoints all 21 `defect~N` citations at their neighbours. That is precisely how
    `entry~14` became wrong for three review passes while every gate stayed green.

    So the identity of each index is pinned here, and every citation is resolved against the table.
    """
    rows = _defect_rows()
    assert sorted(rows) == list(range(1, len(rows) + 1)), (
        f"tab:defects rows are {sorted(rows)}, not contiguous from 1"
    )
    assert set(_DEFECT_IDENTITY) == set(rows), (
        f"tab:defects holds {sorted(rows)} but this guard pins {sorted(_DEFECT_IDENTITY)}; a "
        "defect was added or removed, so every `defect~N` citation needs re-reading before the "
        "pin is updated"
    )
    for n, phrase in _DEFECT_IDENTITY.items():
        assert phrase in rows[n], (
            f"defect~{n} now reads {rows[n][:90]!r}, which does not contain {phrase!r}. The rows "
            "were reordered or renumbered; the series cites them by index, so the citations moved "
            "too."
        )

    cited = 0
    sources = [PAPERS / s / "main.tex" for s in SLUGS] + [PAPERS / "shared/loop.tex"]
    for path in sources:
        flat = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
        for m in re.finditer(r"defects?~(\d+)(?:--(\d+))?", flat):
            for n in (int(g) for g in m.groups() if g):
                assert n in rows, (
                    f"{path.name} cites defect~{n}, but the table holds {len(rows)} rows"
                )
                cited += 1
    assert cited >= 21, (
        f"found {cited} defect citations, expected at least 21; this guard has stopped covering "
        "the sites it was built for"
    )


def _direction_codes_defined() -> set[str]:
    """The direction codes Paper I's list defines, by number. D1 and D2 carry a prime."""
    text = (PAPERS / "p1-conditional-lever/main.tex").read_text(encoding="utf-8")
    start = text.index("Each entry below changes one variable")
    body = text[start : text.index("\\end{description}", start)]
    return set(re.findall(r"\\item\[D(\d{1,2})", body))


def test_every_direction_code_the_series_cites_is_defined() -> None:
    """The third positional pointer in this series, after register entries and defect numbers.

    D6 was registered with a success threshold, bought, and then deleted from the list in the
    same commit that bought it. Nothing noticed for the whole life of the split, because D6 was
    never cited by name: only the declared count of bought directions was wrong, and that count
    is now derived in `verify_paper_claims`. This guard covers the other direction, a code cited
    with no entry behind it, which is what deleting D4 or D7 would produce -- both are cited
    from prose, and D3 is cited from Paper III as well.
    """
    defined = _direction_codes_defined()
    assert defined, "Paper I's directions list defines no D-codes; the list or its header moved"
    cited = 0
    for slug in SLUGS:
        flat = re.sub(r"\s+", " ", (PAPERS / slug / "main.tex").read_text(encoding="utf-8"))
        for m in re.finditer(r"(?<![A-Za-z\\])D(\d{1,2})(?![\d.])", flat):
            where = flat[max(0, m.start() - 70) : m.start()]
            assert m.group(1) in defined, (
                f"{slug} mentions D{m.group(1)} after ...{where!r}, but Paper I's directions "
                f"list defines {sorted(defined)}"
            )
            cited += 1
    assert cited >= 24, (
        f"found {cited} direction-code mentions, expected at least 24; this guard has stopped "
        "covering the sites it was built for"
    )


#: Term -> (regex that matches its DEFINITION, regex that matches a bare USE). A paper is
#: main.tex plus three shared inputs, so both questions have to be asked against the expanded
#: document. Asking them against main.tex alone is how a reviewer of this series twice concluded
#: that an already-defined symbol was undefined, and "fixed" it by defining it a second time.
_INTRODUCED_BEFORE_USE = {
    "genome": (r"the \\emph\{genome\} is", r"\bgenome\b"),
    "gene": (r"a \\emph\{gene\} is", r"\bgenes?\b"),
    "p_0": (r"\$p_0\$, the base rate", r"p_0"),
    "R": (r"refine\s+budget of \$R\$", r"\$R\{[:=]\}"),
    # A GEPA parameter, printed once in each of Papers I and III and glossed in neither. Paper I
    # listed it as one of the study's three corrections beside two that carried a defect
    # attribution; Paper III named it only in a corrections-register entry, twenty pages after
    # describing the selector it configures without ever tying the two together.
    "module_selector": (
        r"component selector \(\\texttt\{module\\_selector\}\)"
        r"|\\texttt\{module\\_selector=all\} \(a proposal now rewrites every gene",
        r"module\\_selector",
    ),
}


def _expanded(slug: str) -> str:
    """One paper as a reader meets it: main.tex with its shared inputs inlined in place."""
    text = (PAPERS / slug / "main.tex").read_text(encoding="utf-8")

    def inline(m: re.Match[str]) -> str:
        return (PAPERS / "shared" / f"{m.group(1)}.tex").read_text(encoding="utf-8")

    return re.sub(r"\\input\{\.\./shared/(\w+)\}", inline, text)


def test_every_coined_term_is_introduced_before_a_paper_uses_it() -> None:
    """Each paper has to stand alone, and the terms it stands on are its own coinage.

    "Genome" ran to 116 uses across the series, in two of the three abstracts, defined in no
    paper and in neither shared input: the central noun of the work, and a reader of any one
    paper met it cold. Nothing caught that, because every check here asks whether a number traces
    to an artifact and none asked whether a word had been introduced.

    The expansion matters as much as the check. These papers are main.tex plus shared/preamble,
    shared/series and shared/loop, and a term defined in a shared input is defined in all three
    papers at the point that input appears. Two separate audits of this series read main.tex
    alone, concluded that p_0 and R were undefined, and added redundant second definitions.
    """
    for slug in SLUGS:
        whole = re.sub(r"\s+", " ", _expanded(slug))
        # The abstract is exempt, as a stated decision rather than an oversight. All three run
        # within 18 characters of arXiv's 1920-character cap, so a gloss does not fit, and an
        # abstract using a term its body defines is ordinary. Every one of them does: Paper I
        # says "evolved genes", Papers II and III say "genome". The body is not exempt.
        text = whole[whole.index("\\end{abstract}") :]
        for term, (define_pat, use_pat) in _INTRODUCED_BEFORE_USE.items():
            use = re.search(use_pat, text)
            if use is None:
                continue
            definition = re.search(define_pat, text)
            assert definition is not None, (
                f"{slug} uses {term!r} and neither it nor its shared inputs define it; a reader "
                "of this paper alone meets the term cold"
            )
            assert definition.start() < use.start(), (
                f"{slug} uses {term!r} at character {use.start()} but defines it at "
                f"{definition.start()}, so the first use comes first"
            )


#: (slug, entry number) -> a phrase that must occur in that register entry. These are the
#: pointers a paper makes into its OWN register, which the cross-paper guard above does not see.
#: They are positional in exactly the same way, and they do not quote the entry they cite, so a
#: keyword census stands in for the quoted phrase that guard relies on.
#: The number is READ from the citing sentence, never hard-coded beside it. A first version of
#: this map keyed on (paper, index) and pinned only what the entry contained, so renumbering the
#: citation from entry~2 to entry~3 changed nothing it looked at and the mutation passed.
_INTRA_REGISTER_POINTERS = {
    ("p1-conditional-lever", r"the depth bound of entry~(\d+)"): "depth bound",
    ("p1-conditional-lever", r"the \$1\.12\$ of entry~(\d+)"): r"1\.12",
    ("p2-repair-not-detection", r"the borrowed denominator entry~(\d+) retired"): "borrowed",
    ("p2-repair-not-detection", r"why entry~(\d+) needed reinstating"): "beats the tool-fed loop",
    ("p2-repair-not-detection", r"the equivalence overclaim entry~(\d+) withdraws"): (
        "statistically indistinguishable"
    ),
    ("p2-repair-not-detection", r"the guard written for entry~(\d+) did not match"): (
        "statistically indistinguishable"
    ),
}


def test_a_papers_pointers_into_its_own_register_still_land() -> None:
    """`entry~14` broke because inserting an entry moves every number after it.

    The guard written for that covers pointers into ANOTHER paper's register, which is where the
    defect was found. It does not cover a paper pointing into its own, and there are four of
    those: two in Paper I and two in Paper II. They are positional in the same way and fail in
    the same silent way, since a number that still resolves to *an* entry breaks no build.

    Entries have been appended to these registers five times during review, which is why the
    four still land; an insertion anywhere but the end would have moved them.
    """
    for (slug, citing), phrase in _INTRA_REGISTER_POINTERS.items():
        flat_paper = re.sub(r"\s+", " ", (PAPERS / slug / "main.tex").read_text(encoding="utf-8"))
        cite = re.search(citing, flat_paper)
        assert cite is not None, (
            f"{slug} no longer carries the citing sentence {citing!r}; this guard reads the entry "
            "number out of that sentence, so a rewording hides the pointer rather than moving it"
        )
        n = int(cite.group(1))
        entries = _register_entries(slug)
        assert 1 <= n <= len(entries), (
            f"{slug} cites its own entry~{n}, but its register holds {len(entries)}"
        )
        body = re.sub(r"\s+", " ", entries[n - 1])
        assert re.search(phrase, body, re.I), (
            f"{slug}'s entry~{n} reads {body[:110]!r}, which does not match {phrase!r}. An entry "
            "was probably inserted above it, moving every pointer after it."
        )

    found = 0
    for slug in SLUGS:
        flat = re.sub(r"\s+", " ", (PAPERS / slug / "main.tex").read_text(encoding="utf-8"))
        for m in re.finditer(r"entry~(\d+)", flat):
            window = flat[max(0, m.start() - 95) : m.end() + 45]
            if re.search(r"Paper~I{1,3}|companion paper", window):
                continue  # a cross-paper pointer, covered by the guard above
            found += 1
    assert found == len(_INTRA_REGISTER_POINTERS), (
        f"found {found} pointers into a paper's own register, but {len(_INTRA_REGISTER_POINTERS)} "
        "are pinned; enrol the new one so it cannot drift unwatched"
    )


def test_the_bibliography_has_no_latex_special_characters_that_eat_text() -> None:
    """BibTeX copies a field verbatim into `main.bbl`, which LaTeX then reads as source.

    So an unescaped `%` inside a field value is a comment: every character after it on that line
    disappears from the printed reference list, silently, with no warning in any log. Four notes
    carried one, and all three shipped PDFs rendered "by 6fewer rollouts", "by over
    10AIME-2025)" and "often 5null is read against" for the whole life of the bibliography. A
    fifth entry had escaped its percent signs correctly, which is what made the omission a slip
    rather than a convention.

    `~` is checked too: it is a non-breaking space, not a tilde, so "~20%" prints as "20%" and an
    approximation in a note marked "verified" silently becomes an exact figure.
    """
    lines = (PAPERS / "shared" / "references.bib").read_text(encoding="utf-8").splitlines()
    inside, offenders = False, []
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("@"):
            inside = True
        if inside:
            for char, why in (("%", "comments out the rest of the line"), ("~", "is a space")):
                if re.search(rf"(?<!\\){re.escape(char)}", line):
                    offenders.append(f"line {n}: {char!r} {why}: {line.strip()[:90]}")
                    break
            if line.strip() == "}":
                inside = False
    assert not offenders, (
        "unescaped LaTeX special characters inside bibliography entries:\n  "
        + "\n  ".join(offenders)
    )


#: The split proof's size, restated in two documents and derived in neither. It read 322 against a
#: script printing 318 -- and unlike the counts above it had no guard at all, in either document,
#: because the number lives in `verify_split_coverage` rather than in a file listing.
SPLIT_QUANTITY_PATTERNS = (
    ("docs/papers/README.md", r"all (\d+) quantities in the frozen pre-split draft"),
    ("docs/EXPERIMENTS.md", r"# (\d+) pre-split quantities accounted for"),
    ("docs/EXPERIMENTS.md", r"all (\d+) quantities in the pre-split draft"),
    # The fourth site, enrolled 2026-09-03. The comment above records fixing 322 -> 318 "in two
    # documents", and this guard was written over exactly those two: REPRODUCE.md kept saying 322
    # for as long as the guard existed, and the guard passed the whole time. Fixing every site a
    # number occurs at is a different job from listing every site the guard reads, and only the
    # second one is what makes the next drift fail.
    ("docs/papers/REPRODUCE.md", r"All (\d+) are accounted for and none is dropped"),
)


def test_advertised_pre_split_quantity_count_is_the_measured_one() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_split_coverage as vsc

    actual = len(vsc.quantities(vsc.frozen_tex()))
    assert actual > 100, "the frozen draft yielded almost no quantities; the guard would be empty"
    for rel, pattern in SPLIT_QUANTITY_PATTERNS:
        m = re.search(pattern, (ROOT / rel).read_text(encoding="utf-8"))
        assert m, (
            f"{rel} no longer states the pre-split quantity count in the form this guard reads. "
            "Restore the sentence or update SPLIT_QUANTITY_PATTERNS -- do not delete the guard."
        )
        assert int(m.group(1)) == actual, (
            f"{rel} advertises {m.group(1)} pre-split quantities, the frozen draft yields {actual}"
        )


#: Every document that states how much of the release replays. Three numbers, one registry.
#: `docs/EXPERIMENTS.md` printed "13 verified, 12 skipped" while the script printed 14 and 12 --
#: a comment beside the command whose output contradicts it, which is the worst place for a stale
#: number because a reader checks it by eye against a run they have just done.
REPLAY_COUNT_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "docs/EXPERIMENTS.md": (
        ("replayed", r"# (\d+) verified, \d+ skipped as superseded"),
        ("superseded", r"# \d+ verified, (\d+) skipped as superseded"),
    ),
    "docs/papers/README.md": (
        ("replayed", r"\*\*(\d+)\*\* replay bit-for-bit"),
        ("superseded", r"\*\*(\d+)\*\* are registered with frozen CLI args and labelled superse"),
        ("unreplayable", r"\*\*(\d+)\*\* BigCodeBench runs are listed as\s+unreplayable"),
    ),
    # Added 2026-08-13. This file states the same three counts and no test read it, so when the
    # registry moved from 5/12 to 25/13 the two guarded documents were updated and this one was
    # not. It is the reproduction instructions, which makes it the document a reader most likely
    # runs against -- it told them to expect `ALL 5 RUNS` from a script that prints 25.
    "docs/papers/REPRODUCE.md": (
        ("replayed", r"should print `ALL (\d+) RUNS REPRODUCE FROM CASSETTE`"),
        ("superseded", r"A further \*\*(\d+)\*\* runs are reported as skipped as superseded"),
        ("unreplayable", r"The remaining \*\*(\d+)\*\* of the 43 released runs"),
    ),
}


def test_advertised_replay_counts_equal_the_registry() -> None:
    """The replay guarantee's size, wherever a document states it, against the registry itself.

    Not against a literal: `verify_reproducibility` is imported and its lists are counted, so the
    advertisement and the thing advertised cannot part company. The three counts must also sum to
    the release, which is the property that makes the guarantee meaningful -- a partition that
    does not cover everything is a guarantee with an unstated remainder.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import verify_reproducibility as vr

    actual = {
        "replayed": len({label for label, *_ in vr.RUNS} - set(vr.SUPERSEDED)),
        "superseded": len(vr.SUPERSEDED),
        "unreplayable": len(vr.UNREPLAYABLE),
    }
    released = len(list((ROOT / "docs").glob("validation-study*.json")))
    assert sum(actual.values()) == released, (
        f"the three replay states cover {sum(actual.values())} runs but the release holds "
        f"{released}; the partition has a remainder nothing accounts for"
    )

    seen = 0
    for rel, patterns in REPLAY_COUNT_PATTERNS.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for key, pattern in patterns:
            m = re.search(pattern, text)
            assert m, (
                f"{rel} no longer states its {key} count in the form this guard reads. Restore "
                "the sentence or update REPLAY_COUNT_PATTERNS -- a guard that stops matching "
                "does not fail, it vanishes, which is how six checks once disappeared silently."
            )
            assert int(m.group(1)) == actual[key], (
                f"{rel} says {m.group(1)} {key} run(s), the registry has {actual[key]}"
            )
            seen += 1
    assert seen == sum(len(p) for p in REPLAY_COUNT_PATTERNS.values()) >= 5


def test_the_ceiling_override_count_agrees_everywhere_it_is_stated():
    r"""The `$5` default was raised by five runs. Three documents and two papers say so.

    `verify_paper_claims.py` checks every site inside the papers, but its corpus is
    `paper_sources.all_tex()`, so a restatement in Markdown is outside it. `docs/EXPERIMENTS.md`
    said "Four runs raised it explicitly" until 2026-08-13 -- the fifth site of one fact, and the
    second to carry the wrong number after Paper III's.

    The count comes from the release, and the sites are found rather than listed, so a new
    restatement is covered the moment it is written in either of these two phrasings.
    """
    over = [
        est
        for path in sorted((ROOT / "docs").glob("validation-study*.json"))
        for est in [json.loads(path.read_text(encoding="utf-8"))["meta"].get("estimated_usd")]
        if est is not None and est > 5
    ]
    words = {"four": 4, "five": 5, "six": 6, "seven": 7, "three": 3, "two": 2}
    sites = []
    for rel in ("docs/EXPERIMENTS.md", "docs/papers/README.md", "docs/papers/REPRODUCE.md"):
        path = ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(\w+) runs? (?:raised it explicitly|overrode)", text):
            word = m.group(1).lower()
            assert word in words, f"{rel}: unrecognised count word {word!r} in {m.group(0)!r}"
            sites.append((rel, m.group(0), words[word]))
    assert sites, (
        "no document states the ceiling-override count in a form this guard reads; it was "
        "stated in three, and a guard that stops matching vanishes rather than failing"
    )
    wrong = [(rel, phrase, n) for rel, phrase, n in sites if n != len(over)]
    assert not wrong, f"{len(over)} runs exceeded the default; these say otherwise: {wrong}"


def test_the_vacuous_verifier_count_agrees_in_the_markdown_too() -> None:
    r"""Six of eleven IFBench verifiers pass on the empty string. Two papers and one doc say so.

    `verify_paper_claims.py` now sweeps every site inside the papers, but its corpus is
    `paper_sources.all_tex()`, so a restatement in Markdown is outside it -- the same corpus gap
    that let `docs/EXPERIMENTS.md` carry the wrong ceiling-override count.
    `docs/validation-study-summary.md` carried the withdrawn `five` from the split until
    2026-08-16, the third site of one fact and the second to be corrected in the papers alone.

    The count is derived by calling the verifiers, and the sites are found rather than listed, so
    a new restatement is covered the moment it is written. `PRE-SPLIT-SOURCE.tex` is excluded by
    construction: it is the frozen pre-split draft, never built, and its `five` is the historical
    record the register cites.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from whystuck.eval.ifbench import VERIFIERS

    vacuous = sum(1 for fn in VERIFIERS.values() if fn(""))
    words = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

    sites = []
    for path in sorted(ROOT.glob("docs/**/*.md")):
        text = path.read_text(encoding="utf-8")
        # `[\s>]+` because the sentence wraps inside a blockquote, so the continuation carries a
        # `>` marker: the Markdown twin of never reflowing a LaTeX line that contains a number.
        for m in re.finditer(r"(\w+) of eleven[\s>]+IFBench verifiers", text):
            word = m.group(1).lower()
            assert word in words, f"{path.name}: unrecognised count word {word!r}"
            sites.append((path.name, m.group(0), words[word]))
    assert sites, (
        "no Markdown document states the vacuous-verifier count in a form this guard reads; it "
        "was stated in one, and a guard that stops matching vanishes rather than failing"
    )
    wrong = [(f, phrase, n) for f, phrase, n in sites if n != vacuous]
    assert not wrong, (
        f"{vacuous} of {len(VERIFIERS)} verifiers pass on the empty string; these say "
        f"otherwise: {wrong}"
    )


#: Documents that together are expected to account for the release in prose. Not every document:
#: the point is that a released run is explained SOMEWHERE a reader will look, not that each file
#: mentions all of them.
_PROSE_DOCS = (
    "docs/EXPERIMENTS.md",
    "docs/papers/README.md",
    "docs/papers/REPRODUCE.md",
    "docs/validation-study-summary.md",
    "README.md",
)


def test_every_released_run_is_named_in_prose():
    """A released run nobody explains is a published artifact with no account of what it is.

    Five were in that state until 2026-08-13: the budget sweep's `mmc20/40/56/80/160`. They carry
    committed JSON and cassettes, they replay, and Paper I's budget table is built from them, so
    nothing was wrong with the data -- but no prose document named them, and they are also the five
    runs with no generated `.md` report, so there was no page anywhere describing them.

    A run's own generated report counts as naming it: that is a document a reader can find. The
    check is therefore "named in some prose document OR has its own report", and the five had
    neither.
    """
    docs = "\n".join(
        (ROOT / rel).read_text(encoding="utf-8") for rel in _PROSE_DOCS if (ROOT / rel).is_file()
    )
    unexplained = []
    for path in sorted((ROOT / "docs").glob("validation-study*.json")):
        stem = path.stem  # e.g. validation-study-mmc20
        slug = stem.removeprefix("validation-study").lstrip("-") or "baseline"
        if path.with_suffix(".md").is_file():
            continue
        if slug != "baseline" and slug not in docs:
            unexplained.append(path.name)
    assert not unexplained, (
        "released runs with no generated report and named in no prose document: "
        f"{unexplained}. Either name them in one of {list(_PROSE_DOCS)} or generate their report; "
        "a run cited only by a LaTeX table is one a reader cannot look up."
    )


#: Every way a reader can learn what an arm letter means, in the order they resolve it. A paper
#: satisfies the guard below by carrying one of these BEFORE its first use of the letter. The list
#: is deliberately not keyed by paper: adding a fourth paper needs no entry here, and a paper that
#: drops its definition cannot be rescued by an entry someone wrote for it once.
_ARM_RESOLUTIONS = (
    # The four-arms block, which Papers I and III both carry in full.
    r"\\item \\textbf\{%s\}",
    # An arm glossed where it is used, which is how Paper II introduces the only arm it measures.
    r"arm~\\textbf\{%s\} of the underlying study: a fixed genome, evolved by nothing",
    # An attributed pointer at the paper that does define it. Paper II measures no B or D and
    # says so, which is a better answer than repeating a definition it never uses.
    r"\\textbf\{%s\}(?:[^.]{0,80}?)defined in Paper~III",
    r"\\textbf\{B\} and \\textbf\{%s\}[^.]{0,80}?defined in Paper~III",
)


def test_no_paper_uses_an_arm_letter_before_a_reader_can_resolve_it() -> None:
    """A, B, C, D are the series' oldest abbreviation and the least glossed.

    The four-arms block sits in each paper's method section, and both Paper I's introduction and
    Paper III's contributions list reached back for `arm~D` before it -- 61 lines early in one,
    115 in the other. Nothing caught it: the coined-term guard above checks the nouns this work
    invented, and an arm letter is not a noun. Published style guidance is unanimous and simple
    here (define at first use, then abbreviate), and the failure is the ordinary one it exists to
    prevent, in a series where the definition lives in a different paper for one of the three.

    The abstract is exempt on the same stated grounds as the coined-term guard, and none of the
    three abstracts uses a letter anyway.
    """
    for slug in SLUGS:
        # Collapsed, because two of the resolutions below are sentences that wrap in the source
        # and a guard that reads raw lines would miss them and report a definition as absent.
        whole = re.sub(r"\s+", " ", _expanded(slug))
        text = whole[whole.index("\\end{abstract}") :]
        for letter in "ABCD":
            # One `arms?` can carry several letters ("arms B and D", "arms~B and~D"), and the
            # separator varies (tilde, space, hyphen). A first version required the letter to
            # follow `arms?` with nothing between, so it read none of Paper II's B and D uses
            # and reported the paper clean; the mutation that deleted its pointer passed.
            letters = r"(?:\\textbf\{[ABCD]\}|\b[ABCD]\b)"
            uses = [
                m.span()
                for m in re.finditer(
                    rf"\barms?[~ \-]*{letters}(?:\s*(?:and|,|--|/)[~ ]*{letters})*", text
                )
                if re.search(rf"\\textbf\{{{letter}\}}|\b{letter}\b", m.group(0))
            ]
            if not uses:
                continue
            resolutions = [
                m.start()
                for pat in _ARM_RESOLUTIONS
                if (m := re.search(pat % letter, text)) is not None
            ]
            assert resolutions, (
                f"{slug} uses arm {letter} and never defines it, glosses it, or says which "
                "paper does; a reader of this paper alone meets the letter cold"
            )
            # Compared against the END of the first use, not its start. Paper II resolves B and
            # D inside the sentence that first names them ("arms B and D (prompt-only and
            # loop-level, defined in Paper III)"), so the resolution begins five characters
            # after the use does and a start-to-start comparison called that a forward
            # reference. What a reader needs is to meet the resolution before finishing the
            # phrase, which is what this asks.
            first_start, first_end = min(uses)
            assert min(resolutions) <= first_end, (
                f"{slug} uses arm {letter} at character {first_start} but a reader cannot "
                f"resolve it until {min(resolutions)}"
            )


def _cited_runs(text: str) -> set[int]:
    """Every numbered run a paper cites, ranges expanded to their members."""
    out: set[int] = set()
    for lo, hi in re.findall(r"runs?~? ?(\d+)(?:--(\d+))?", text):
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _tabulated_runs(text: str) -> set[int]:
    """Run numbers a paper lists itself, read out of any table that carries a run column.

    A run column is found by its header, not by position, and the values are taken from the
    matching cell of every body row, so a table gaining a column does not shift what this reads.
    """
    out: set[int] = set()
    for block in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S):
        stripped = re.sub(
            r"\\(?:begin|end)\{tabular\}(?:\{[^}]*\})?|\\(?:top|mid|bottom)rule", " ", block
        )
        rows = [[c.strip() for c in r.split("&")] for r in stripped.split(r"\\")]
        head = next(
            ((i, r) for i, r in enumerate(rows) if any(re.fullmatch(r"(?i)run", c) for c in r)),
            None,
        )
        if head is None:
            continue
        idx = next(i for i, c in enumerate(head[1]) if re.fullmatch(r"(?i)run", c))
        for row in rows[head[0] + 1 :]:
            if len(row) == len(head[1]) and re.fullmatch(r"\d+", row[idx]):
                out.add(int(row[idx]))
    return out


#: What each paper cites by number. Asserted, not merely used: a new citation has to be declared
#: here, which is the half a presence-only check would leave unread.
_RUN_CITATIONS = {
    "p1-conditional-lever": {2, 3, 4, 5, 6, 7, 8, 9, 10},
    "p2-repair-not-detection": {3, 4, 5, 7, 8, 9, 10},
    "p3-silent-no-op": {1, 3, 6, 7, 8, 9, 10},
}

#: A sentence saying where the run numbering comes from, paired with what that sentence commits
#: the paper to about its OWN tables. Each paper carries one, and each says something different:
#: Paper I tabulates no run at all, Papers II and III both point at a table of their own. The
#: commitment is what makes the table-reading half of the guard load-bearing -- a first version
#: took the union of the local table and Paper III's, and since Paper III lists all ten, no
#: paper's own table was ever read.
#: Where the numbering itself comes from. Kept separate from the commitment below because a
#: mutation that deleted this pointer and left the commitment behind passed: any one sentence
#: satisfying a single "says something about runs" check is not the same as saying both things.
_RUN_NUMBERING_POINTER = (
    r"numbered as in the companion Paper~III's table of the original runs",
    r"listed with the rest of the numbering in Table~\\ref\{tab:results\}",
)

#: What the paper commits to about its own tables.
_RUN_TABLE_COMMITMENT = (
    (r"This paper tabulates none of them", "none"),
    (r"every one this paper cites carries its own row in Table~", "all"),
    (r"listed with the rest of the numbering in Table~\\ref\{tab:results\}", "all"),
)


def test_every_numbered_run_a_paper_cites_is_one_a_reader_can_look_up() -> None:
    """`run~3` is an index into a release, and an index is worth nothing without its list.

    Paper III numbers the ten original runs in a table and points at it. Paper II cites seven
    numbers and lists all seven in its own detector table. Paper I cites four, tabulates none of
    them, and had no sentence saying what the numbering indexes -- so its reader met `run~4` and
    `run~6` as bare labels, in a paper that has no list of runs at all.

    Both halves are read here. The tabulated set comes from the papers' own tables and the cited
    set from their prose, so the guard fails if a citation appears that no table carries and
    fails again if the numbering statement goes missing.
    """
    listed = {slug: _tabulated_runs(_expanded(slug)) for slug in SLUGS}
    for slug in SLUGS:
        text = re.sub(r"\s+", " ", _expanded(slug))
        cited = _cited_runs(text)
        assert cited == _RUN_CITATIONS[slug], (
            f"{slug} cites runs {sorted(cited)}, declared {sorted(_RUN_CITATIONS[slug])}; a new "
            "numbered run has to be declared so this guard reads it"
        )
        assert any(re.search(pat, text) for pat in _RUN_NUMBERING_POINTER), (
            f"{slug} cites numbered runs and never says where the numbering comes from"
        )
        commitments = [c for pat, c in _RUN_TABLE_COMMITMENT if re.search(pat, text)]
        # The commitment is derived from the tables rather than declared beside the paper, so
        # both directions bite: dropping the sentence fails, and so does gaining a run table
        # without saying so.
        if listed[slug]:
            assert "all" in commitments, (
                f"{slug} tabulates runs {sorted(listed[slug])} and never tells a reader that "
                "the runs it cites are the ones listed there"
            )
            assert cited <= listed[slug], (
                f"{slug} says every run it cites has a row of its own and runs "
                f"{sorted(cited - listed[slug])} have none"
            )
        else:
            assert "none" in commitments, (
                f"{slug} tabulates no numbered run and does not say so, so a reader looking for "
                "the list has nothing to find and no notice of it"
            )
        # Whatever a paper says about its own tables, the number has to be findable somewhere in
        # the series, and Paper III is where the numbering lives.
        resolvable = listed[slug] | listed["p3-silent-no-op"]
        assert cited <= resolvable, (
            f"{slug} cites runs {sorted(cited - resolvable)} which no table in the series lists"
        )


#: The two modules that decide what the loop emits. A monospace all-caps token in a paper is a
#: value a reader should be able to find here, and only these files settle that: `FAIL` occurs in
#: `bcb_exec.py` as a unittest label, which has nothing to do with a critique's verdict, so a
#: whole-tree grep would have accepted the token this guard was written to catch.
_LOOP_SOURCE = ("src/whystuck/eval/llm_worker.py", "src/whystuck/eval/adapter.py")


def _texttt(slug: str) -> set[str]:
    return {m.replace("\\_", "_") for m in re.findall(r"\\texttt\{([^}]*)\}", _expanded(slug))}


def test_a_monospace_verdict_token_is_one_the_loop_actually_emits() -> None:
    """`FIX` and `FAIL` were typeset as protocol tokens and the harness emits neither.

    The critique's contract is one line of `llm_worker.py`: "Reply exactly PASS if every
    requirement is satisfied; otherwise list what to fix." So `PASS` is a token and the failure
    branch is prose. Paper I printed `\\texttt{FIX}` five times, four as a coined label and once
    as a claim that a detector "emits FIX unconditionally"; Paper II paired `PASS` with a
    `FAIL` that exists nowhere in the loop. Both read as greppable identifiers, which in this
    series is a promise: every other monospace token in these papers resolves to something in
    the release.
    """
    emitted = "\n".join((ROOT / f).read_text(encoding="utf-8") for f in _LOOP_SOURCE)
    for slug in SLUGS:
        for token in sorted(_texttt(slug)):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,15}", token):
                continue
            assert token in emitted, (
                f"{slug} prints \\texttt{{{token}}} as a value the loop emits and no loop "
                f"module contains it; the critique's own contract lives in {_LOOP_SOURCE[0]}"
            )


def _released_slugs() -> set[str]:
    return {
        p.name[len("validation-study-") : -len(".json")]
        for p in (ROOT / "docs").glob("validation-study-*.json")
    }


def test_a_printed_artifact_name_is_the_name_the_release_uses() -> None:
    """Paper III printed `\\texttt{c2}` as `\\texttt{C2}`, twice, in its corrections register.

    The register exists so a reader can check a withdrawn claim against the artifact, and the
    entry pointing them at it named a file that does not exist: the release holds
    `validation-study-c2.json` and a case-sensitive filesystem does not care that the paper
    meant it. This compares case-folded against exact, so a token that looks like a released
    study has to be spelled the way the release spells it.
    """
    released = _released_slugs()
    assert released, "no released studies found; this guard would pass vacuously"
    for slug in SLUGS:
        printed = {t for t in _texttt(slug) if t.lower() in released}
        wrong = sorted(t for t in printed if t not in released)
        assert not wrong, (
            f"{slug} prints {wrong} and the release spells them {sorted(t.lower() for t in wrong)}"
        )


def test_an_experiment_code_is_bound_to_the_artifacts_it_names() -> None:
    """`C1` appeared three times in Paper I and was defined nowhere in it.

    It is the internal name of the hand-written-gene ablation, which the release publishes as
    `validation-study-c1-base` and `validation-study-c1-hand`. The prose glossed the finding
    and the table used the bare code, so a reader could follow the argument and still not know
    which two files to open. Any code of this shape has to sit in a paper that also prints one
    of the released names it stands for.
    """
    released = _released_slugs()
    for slug in SLUGS:
        text = re.sub(r"\s+", " ", _expanded(slug))
        printed = _texttt(slug)
        for code in sorted(set(re.findall(r"\bC(\d)\b", text))):
            bound = {s for s in released if s.startswith(f"c{code}")} & printed
            assert bound, (
                f"{slug} uses the experiment code C{code} and prints none of the released "
                f"artifacts it names ({sorted(s for s in released if s.startswith(f'c{code}'))})"
            )


#: A named system and the bibliography entry that introduces it. Only systems whose name is a
#: proper noun a reader would look up, not every cited work.
_NAMED_SYSTEMS = {
    "GEPA": "agrawal2025gepa",
    "IFEval": "zhou2023ifeval",
    "BigCodeBench": "zhuo2024bigcodebench",
    "IFBench": "pyatkin2025ifbench",
    "OPRO": "yang2023opro",
    "DSPy": "khattab2023dspy",
    "LiveCodeBench": "jain2024livecodebench",
}


def test_a_paper_that_names_a_system_cites_it() -> None:
    """Paper II named GEPA three times and cited it nowhere; Paper III did the same to
    BigCodeBench.

    Both defects lived in the shared inputs, which is why seventeen passes went past them. The
    series note says the gene/genome vocabulary is ours rather than GEPA's, and the harness
    description lists the BigCodeBench runs as unreplayable -- so every paper says those names,
    while the citation sat in the two main.tex files that happened to discuss the system at
    length. bibtex prints only what a paper cites, so Paper II's reference list had no GEPA in
    it at all: the optimizer the whole series is about, named in a paper that never told a
    reader where to find it.

    Read against the expansion, because that is the document a reader is handed, and the
    citation for a name introduced in a shared input has to live in that input.
    """
    for slug in SLUGS:
        text = _expanded(slug)
        cited = {
            key.strip()
            for group in re.findall(r"\\cite[tp]?\{([^}]*)\}", text)
            for key in group.split(",")
        }
        for name, key in _NAMED_SYSTEMS.items():
            if not re.search(rf"\b{name}\b", text):
                continue
            assert key in cited, (
                f"{slug} names {name} and never cites {key}, so the name does not appear in "
                "this paper's reference list at all"
            )


def _paper_source_paths(slug: str) -> list[str]:
    return [f"docs/papers/{slug}/main.tex", "docs/papers/shared"]


def _last_source_month(slug: str) -> str | None:
    """`YYYY-MM` of the newest change to this paper's sources, uncommitted edits included."""
    paths = _paper_source_paths(slug)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0:
        return None
    if dirty.stdout.strip():
        # Being edited now, so now is when it was last revised. Without this the guard reads the
        # previous commit and lets the very edit that outdates the line pass.
        return _dt.date.today().strftime("%Y-%m")
    logged = subprocess.run(
        ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return logged.stdout.strip() or None


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def test_the_date_on_the_title_page_is_not_older_than_the_paper() -> None:
    """All three said `July 2026` on 5 September, through eighteen review passes.

    July is when the split happened. Every pass since has rewritten prose, withdrawn claims and
    added register entries, and the title page kept advertising a draft five weeks older than
    the document under it -- the one number on the front page, and the only one in these papers
    that no check ever read.

    Compared against the sources' own history rather than against a hard-coded month, so the
    line has to move when the papers do and this test does not have to be edited every time.
    """
    for slug in SLUGS:
        m = re.search(r"\\date\{(\w+) (\d{4})\}", (PAPERS / slug / "main.tex").read_text())
        assert m, f"{slug} has no parseable \\date"
        stated_month, stated_year = m.group(1), int(m.group(2))
        assert stated_month in _MONTHS, f"{slug} dates itself {stated_month!r}"
        stated = f"{stated_year:04d}-{_MONTHS.index(stated_month) + 1:02d}"
        revised = _last_source_month(slug)
        if revised is None:
            pytest.skip("no git history available to date the sources against")
        assert stated >= revised, (
            f"{slug} says {stated_month} {stated_year} on its title page and its sources were "
            f"last revised {revised}"
        )


#: What counts as saying WHICH companion. Either the series note's own numbering, or a phrase
#: naming the paper's subject -- both resolve against the note every paper carries.
_COMPANION_DISAMBIGUATORS = (
    r"Paper~(?:I|II|III)\b",
    r"conditional[- ]lever",
    r"ceiling paper",
    r"loop's ceiling",
    r"silent no-op",
    r"detector result",
    # Plural, meaning both of them, which needs no further identification.
    r"companion papers",
)


def test_a_companion_reference_says_which_companion() -> None:
    """Each paper has TWO companions, and "the companion paper" names neither.

    Paper I refers to its companions by number fifty-two times and is never ambiguous. Paper III
    used the bare phrase nine times, for both companions: `the companion paper's $p_0$ axis` is
    Paper I and `the companion ceiling paper` is Paper II, in the same document, with nothing
    telling a reader that the first phrase excludes the second. Paper II had three.

    This matters more here than in an ordinary cross-reference because these are the sentences
    that hand a claim to another paper. A reader who cannot tell which paper holds the evidence
    cannot go and check it, which is the whole point of splitting the work and saying so.
    """
    for slug in SLUGS:
        # Comment lines dropped first: `shared/series.tex` opens with a comment calling itself
        # the companion-paper note, which a reader never sees and this guard would fail on.
        body = "\n".join(
            ln for ln in _expanded(slug).splitlines() if not ln.lstrip().startswith("%")
        )
        text = re.sub(r"\s+", " ", body)
        for sentence in re.split(r"(?<=[.:;]) ", text):
            # "companion" has to be modifying a paper, within two words of it, so that
            # "the companion ceiling paper" triggers and "Three companion rules cost as
            # little" does not. Requiring both words anywhere in the sentence flagged the
            # second as soon as any nearby clause happened to mention a paper.
            if not re.search(r"\bcompanion(?:[- ]\w+){0,2}[- ]papers?\b", sentence):
                continue
            assert any(re.search(p, sentence) for p in _COMPANION_DISAMBIGUATORS), (
                f"{slug} says {sentence.strip()[:110]!r} and this paper has two companions"
            )


def _sentences(slug: str) -> list[str]:
    """Body sentences, with the corrections register cut off.

    The register's whole job is to quote the claim a paper withdraws, so every guard below would
    fire on the entry recording the very defect it was written for. It is an appendix and
    clearly delimited, so a live claim cannot hide inside it.
    """
    text = _expanded(slug)
    text = text[: text.index(r"\label{sec:corrections}")]
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))
    return re.split(r"(?<=[.:;]) ", re.sub(r"\s+", " ", body))


def test_no_paper_reads_a_failure_to_reject_as_equivalence() -> None:
    """A non-significant paired test does not make two things the same.

    Paper II said "a perfect checker is statistically indistinguishable from one at chance" in
    its abstract, "indistinguishable from a near-random one" in the body, and, worst, that the
    claim was "confirmed rather than merely unrefuted" -- all from t = 0.33 on five replicates,
    with no equivalence margin registered anywhere in the paper. The series knows the standard:
    Paper I's conclusion says of its own null "this is not an equivalence claim", and the
    bibliography carries Lakens on equivalence testing. The paper had even written the
    defensible form once, and then overclaimed in three other places.

    The guard fires on the pattern rather than the word: "indistinguishable" beside a reported
    effect size. Paper I's "indistinguishable on the training signal" is a claim about a
    mechanism and carries no test, so it stays.
    """
    for slug in SLUGS:
        for sentence in _sentences(slug):
            # The CLASS, not one spelling of it. A first version tested for the exact word
            # "indistinguishable", which is the word the first round of fixes happened to
            # remove. It passed while the contributions list said "indistinguishably" (the
            # adverb is not a superstring of the adjective) and the conclusion said "a null
            # confirmed on five replicates rather than merely unrefuted", which never contained
            # the word at all. A sol-medium inspector found both.
            if not re.search(
                r"indistinguishab|equivalent to|confirmed rather than|null confirmed"
                r"|merely unrefuted",
                sentence,
            ):
                continue
            assert not re.search(r"\$t = [\d.]+\$|[\d.]+\\sigma", sentence), (
                f"{slug} calls something indistinguishable beside a test statistic, which is a "
                f"failure to reject and not an equivalence result: {sentence.strip()[:120]!r}"
            )


def test_a_blanket_preregistration_claim_names_the_harness_exceptions() -> None:
    """`shared/loop.tex` names two lines it does NOT claim registration for. A paper that says
    everything was registered has to say that too.

    Paper II's no-multiplicity paragraph read "Each comparison here was specified before the
    runs that answer it" and used registration to argue no family-wise correction is needed.
    The harness section it inputs says the feedback-actionability pair carries a registration
    version control cannot place before the run, and that pair is this paper's best-supported
    effect. The claim and its own methods section were in the same document, four hundred
    lines apart.
    """
    loop = (PAPERS / "shared" / "loop.tex").read_text(encoding="utf-8")
    assert "for them we do not claim it" in loop, (
        "the harness section no longer carves out unregistered lines; this guard reads that "
        "carve-out and would pass vacuously without it"
    )
    for slug in SLUGS:
        for sentence in _sentences(slug):
            if not re.search(r"specified before the runs", sentence):
                continue
            assert "bar the two" in sentence, (
                f"{slug} claims every comparison was specified before its run and does not "
                f"name the exceptions the harness section carves out: {sentence.strip()[:120]!r}"
            )


def test_the_superseded_runs_are_not_called_one_program_without_the_qualifier() -> None:
    """Paper III asserted, in three live places, the claim it withdraws in a fourth.

    "Every Delta in Table 3 is therefore a comparison of a genome against itself" appeared in
    the contributions list, in the defect section, and in the opening of the very section whose
    body says, sixty-four lines later, that the claim "is too strong" because an unchanged
    verify gene does not make the arms the same program: they still differ in the gene they
    share. The table it quantifies over prints two non-zero rows.

    Only the qualified form survives: the arms were the same program FOR THE GENE THAT DEFINES
    THE DIFFERENCE, which is what makes the contrast uninformative without making the genomes
    identical.
    """
    text = re.sub(r"\s+", " ", _expanded("p3-silent-no-op"))
    assert "does not make the arms the same program" in text, (
        "Paper III no longer carries the withdrawal this guard is anchored to"
    )
    for sentence in _sentences("p3-silent-no-op"):
        if not re.search(r"(?i)\b(every|all)\b", sentence):
            continue
        assert not re.search(r"comparison of a genome against itself", sentence), (
            f"{sentence.strip()[:120]!r} quantifies the genome-identity claim over rows the "
            "paper itself shows are not identities"
        )


def _withdrawn_quotes(slug: str) -> list[str]:
    """The claim each corrections entry opens by quoting, normalised for whitespace.

    Every entry in every register begins `\\item \\textbf{``...''}`, so the papers already
    publish their own list of retired phrasings. That list is what this reads.
    """
    text = _expanded(slug)
    register = text[text.index(r"\label{sec:corrections}") :]
    return [
        re.sub(r"\s+", " ", q).strip()
        for q in re.findall(r"\\item \\textbf\{``(.+?)''\}", register, re.S)
    ]


def test_no_body_text_restates_a_claim_its_own_register_withdrew() -> None:
    r"""The register says the body states only the surviving claim. Twice it did not.

    Self-maintaining, and narrower than it looks. The papers publish their retired phrasings in
    the register, one quoted claim per entry, so a correction added later becomes a check on the
    body with nobody writing one. What it catches is a VERBATIM restatement.

    What it does not catch, stated because it was measured rather than guessed. Paper II's body
    said `F$_1$ measures the base rate, not the detector` while its register withdrew
    `$\mathrm{F}_1$ is measuring the base rate, not the detector`. Same claim, two spellings of
    the subscript and two forms of the verb, and this guard runs clean over it: replayed against
    the commit where that pair coexisted, it reports zero.

    Loosening it does not work either, and that is the useful part. Normalising LaTeX, stemming
    and matching the quote's content words in order within a bounded span does flag that pair --
    and also flags six sentences on a tree known to be clean, because almost every fix here
    inserts a qualifier: `specified before the runs that answer it` becomes `specified before
    the runs that answer it, bar the two the harness section names`, and an order-with-gaps match
    cannot tell a claim from the same claim scoped. The distinction this class turns on is
    semantic, so the text guard stays narrow and the coverage problem is handled by enumerating
    an error class across all three papers before fixing any site of it.
    """
    for slug in SLUGS:
        whole = _expanded(slug)
        body = re.sub(r"\s+", " ", whole[: whole.index(r"\label{sec:corrections}")])
        quotes = _withdrawn_quotes(slug)
        assert quotes, f"{slug} publishes no quoted withdrawn claims; this guard reads those"
        for quote in quotes:
            assert quote not in body, (
                f"{slug} states in its body a claim its own register withdraws: {quote[:100]!r}"
            )


#: Source lines a float may sit after the first body reference to it. Calibrated, not guessed:
#: measured against a build, Paper I's `tab:repeat` sits 371 lines after its first body reference
#: and lands 8 pages later, and Paper II's `tab:curve` 248 lines and 5 pages, so roughly 46 lines
#: to the page. 600 leaves today's worst case a wide margin and still catches the failure that
#: has cost this project twice: `shared/preamble.tex` records two figures deferred TWENTY-EIGHT
#: pages past their reference, and Paper I ending on two pages carrying nothing but floats.
_MAX_FLOAT_LAG_LINES = 600


def _float_lag(slug: str) -> list[tuple[str, int]]:
    """(label, lines between its first BODY reference and its own definition), worst first.

    Body reference means one outside every float environment. A `\\ref` inside a caption travels
    with its own float, so counting those measures nothing about what a reader is asked to do:
    it reported one table as arriving nine pages BEFORE the thing that pointed at it.
    """
    text = _expanded(slug)
    spans = [
        m.span()
        for env in ("table", "figure")
        for m in re.finditer(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", text, re.S)
    ]
    line_of = lambda pos: text.count("\n", 0, pos) + 1  # noqa: E731
    out = []
    for m in re.finditer(r"\\label\{((?:tab|fig):[^}]+)\}", text):
        label = m.group(1)
        refs = [
            r.start()
            for r in re.finditer(r"\\ref\{" + re.escape(label) + r"\}", text)
            if not any(a <= r.start() < b for a, b in spans)
        ]
        if not refs:
            continue
        lag = line_of(m.start()) - line_of(refs[0])
        if lag > 0:
            out.append((label, lag))
    return sorted(out, key=lambda pair: -pair[1])


def test_no_float_sits_far_past_the_first_reference_to_it() -> None:
    """A reader sent to a table should not have to turn eight pages to reach it.

    Measured before this existed rather than assumed: of 26 floats across the three papers,
    seven sit more than one page from their first body reference, and every one of those turned
    out to be benign. The forward cases cite a table as the provenance of a value they already
    print inline, and the backward cases are floats placed with their own prose and referenced
    again later. None of the 26 is a "see the figure" instruction pointing pages away, so this
    guard was written to hold a line that is currently clean, not to fix something.

    It measures source lines instead of pages so it needs no build and no poppler, which keeps
    it runnable in a clean checkout. The conversion is calibrated against a real build and
    recorded beside the threshold.
    """
    worst: list[str] = []
    for slug in SLUGS:
        lags = _float_lag(slug)
        assert lags, f"{slug}: measured no float lag at all, so this check is vacuous"
        for label, lag in lags:
            if lag > _MAX_FLOAT_LAG_LINES:
                worst.append(f"{slug} {label}: {lag} source lines (~{lag // 46} pages)")
    assert not worst, "floats stranded far past their first reference: " + "; ".join(worst)


#: Jaccard overlap above which two register entries are the same correction written twice.
#: Measured across all ninety-one entries: the closest genuine pair sits at 24% (Paper I's fifth
#: and sixth), then 19% and 15% in the other two. A copy-paste duplicate scores far higher, so
#: 45% separates the two cases with room to spare and does not fire on entries about related
#: defects. The registers grew by twenty-two entries in three days, which is the reason this
#: exists: nothing else would notice the same withdrawal recorded twice.
_MAX_REGISTER_ENTRY_OVERLAP = 0.45

_REGISTER_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "is",
        "are",
        "was",
        "were",
        "to",
        "in",
        "on",
        "and",
        "that",
        "it",
        "its",
        "for",
        "this",
        "by",
        "with",
        "at",
        "as",
        "not",
        "we",
        "our",
        "from",
        "which",
        "be",
        "been",
        "one",
        "two",
    ]
)


def _register_entry_signatures(slug: str) -> list[set[str]]:
    """One content-word signature per corrections entry, LaTeX stripped.

    Named apart from `_register_entries` above on purpose: appending a same-named helper to this
    file shadowed that one, which returns entry TEXT, and the two guards built on it failed with
    a TypeError rather than an assertion. A collision, not a drift.
    """
    text = _expanded(slug)
    register = text[text.index(r"\label{sec:corrections}") :]
    register = register[: register.index(r"\end{enumerate}")]
    out = []
    for item in re.split(r"\n  \\item ", register)[1:]:
        flat = re.sub(r"\\[a-zA-Z]+\*?", " ", item)
        flat = re.sub(r"[^a-zA-Z0-9]+", " ", flat).lower()
        out.append({w for w in flat.split() if w not in _REGISTER_STOPWORDS and len(w) > 2})
    return out


def test_no_correction_is_recorded_in_the_register_twice() -> None:
    """Ninety-one entries across three registers, added over five weeks and often in batches.

    Two entries describing one withdrawal would overstate how much this work found and would
    leave a reader matching a body claim against two different accounts of its correction. The
    signature is content words rather than the quoted claim, because the same correction written
    twice gets quoted two different ways, which is what defeats an exact-text comparison.
    """
    for slug in SLUGS:
        entries = _register_entry_signatures(slug)
        assert len(entries) > 10, f"{slug}: parsed {len(entries)} entries, so this is vacuous"
        assert all(entries), f"{slug}: an entry produced an empty signature"
        for (i, left), (j, right) in itertools.combinations(enumerate(entries, 1), 2):
            overlap = len(left & right) / len(left | right)
            assert overlap <= _MAX_REGISTER_ENTRY_OVERLAP, (
                f"{slug} entries {i} and {j} overlap {overlap:.0%}, which reads as one "
                f"correction recorded twice: {sorted(left & right)[:10]}"
            )
