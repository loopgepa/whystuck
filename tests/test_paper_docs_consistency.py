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
    tex = "\n".join(
        line
        for line in (PAPERS / slug / "main.tex").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("%")
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
