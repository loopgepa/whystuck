"""The arXiv package must carry the file arXiv will not build for you.

arXiv runs pdflatex over the upload and does NOT run BibTeX, so a tarball without `main.bbl` posts
with every citation as a bare `[?]`. `main.bbl` is a build artifact and is gitignored here, which
is exactly what makes omitting it the likely mistake. These tests pin the two properties that
prevent it, plus the plain-text abstract, and none of them needs a LaTeX toolchain: the full
rehearsal lives in the script itself, which fails loudly rather than shipping a broken upload.
"""

from __future__ import annotations

import inspect
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import make_arxiv_package as ap  # noqa: E402
import paper_sources  # noqa: E402


def test_the_bbl_is_staged():
    """Not incidentally: it is the one input arXiv cannot regenerate."""
    assert "main.bbl" in inspect.getsource(ap.stage)


def test_the_rehearsal_never_runs_bibtex():
    """A rehearsal that runs BibTeX would pass on a tarball that posts with [?] citations."""
    src = inspect.getsource(ap.rehearse)
    # The quoted form, so this matches an actual command argument and not the docstring that
    # explains the rule. A first draft asserted on the bare word and failed on its own prose.
    assert '"bibtex"' not in src, "the rehearsal must reproduce arXiv's toolchain, not ours"
    assert '"pdflatex"' in src
    # And the real invocation must exist elsewhere, or the .bbl would never be generated at all.
    assert '"bibtex"' in inspect.getsource(ap.build_bbl)


def test_build_bbl_leaves_the_paper_directory_with_resolved_citations():
    """Three pdflatex passes, not two, and the third must come after bibtex.

    `build_bbl` deletes `main.bbl` first, so pass one runs with no bibliography and writes an aux
    with no `\\bibcite`. Pass two reads that aux, so the PDF it produces prints `[?]` for every
    citation even though it writes the correct aux for a successor. With only two passes the
    packager left `docs/papers/<slug>/main.pdf` at 22 undefined citations and 14 rendered `[?]`,
    which is a broken PDF sitting where a human looks for the finished paper. Source-level because
    the alternative is three LaTeX runs inside the suite.
    """
    src = inspect.getsource(ap.build_bbl)
    calls = re.findall(r'"(pdflatex|bibtex)"', src)
    assert calls.count("pdflatex") >= 3, (
        f"build_bbl runs pdflatex {calls.count('pdflatex')} times; two leaves the paper "
        "directory with unresolved citations"
    )
    assert calls.index("bibtex") < len(calls) - 2, (
        "at least two pdflatex passes must follow bibtex, or the aux it populates is never "
        "typeset from"
    )


def test_the_rehearsal_fails_the_build_rather_than_warning():
    """Undefined citations must be a non-zero exit, not a line of output someone skims past."""
    src = inspect.getsource(ap.main)
    assert "undefined_citations" in src and "return 1" in src


def test_only_the_figures_a_paper_includes_are_staged():
    """Every paper's `figures/` is a symlink to the series' shared directory.

    Copying the tree put all seven of the series' figures in each of the three tarballs, so Paper
    III's package shipped Paper I's forest plot. arXiv ignores unused files, which is why nothing
    caught it; the package was simply carrying two other papers' evidence.
    """
    counts = {}
    for slug in paper_sources.PAPERS:
        names = ap.included_figures(paper_sources.paper_tex(slug))
        counts[slug] = len(names)
        assert names, f"{slug}: no figures found in the source"
        assert len(names) == len(set(names)), f"{slug}: a figure is staged twice"
        for name in names:
            assert (paper_sources.paper_dir(slug) / ap.FIGURES / name).is_file(), (
                f"{slug} includes {name}, which does not exist"
            )
    shared = {p.name for p in (paper_sources.SHARED_DIR / ap.FIGURES).glob("*.pdf")}
    assert sum(counts.values()) == len(shared), (
        f"the three papers include {sum(counts.values())} figures and the shared directory holds "
        f"{len(shared)}; a figure is either unused or included by two papers, and both are worth "
        "looking at rather than shipping"
    )


def test_a_commented_out_include_is_not_staged():
    """The `\\pdfoutput` comment at the top of every paper names `\\includegraphics`.

    Counting that comment is what put SUBMISSION.md's figure column one too high for every paper,
    so the staging pattern is checked against the same trap: it must match the directive and not
    prose about it.
    """
    assert ap.included_figures("% see \\includegraphics{figures/ghost.pdf} for why") == []
    assert ap.included_figures("\\includegraphics[width=\\textwidth]{figures/real.pdf}") == [
        "real.pdf"
    ]


def test_ink_outside_the_text_block_blocks_the_package():
    """Overfull boxes and oversized floats are warnings, and both put ink past the margin.

    Paper I carried a threats table 61.9pt taller than the text block for its whole life. pdflatex
    said so once, in the log, without naming the float; no test read the log and no reader diffs a
    PDF's margins. Both findings are collected by `rehearse` and both gate the tarball.
    """
    findings = inspect.getsource(ap.rehearse)
    assert "Float too large for page" in findings
    assert "Overfull" in findings
    src = inspect.getsource(ap.main)
    gate, tar = src.find("oversized_floats"), src.find("tarfile.open")
    assert gate != -1, "main() no longer reads the oversized-float finding"
    assert "overfull_boxes" in src, "main() no longer reads the overfull-box finding"
    assert gate < tar, "the finding must gate the tarball rather than be printed after it"


def test_plain_abstract_has_no_latex_left():
    """Every macro that resolves against the document must become words; standard math may stay.

    The distinction is what arXiv's metadata field can render. `\\sigma` it renders. `\\dd` is
    defined in our own preamble, which the field never sees, so it would render as nothing at all --
    silently deleting the symbol from a sentence about a measured quantity.
    """
    for slug in paper_sources.PAPERS:
        text = ap.plain_abstract(paper_sources.paper_dir(slug))
        for macro in ("citep", "citet", "emph", "ref{", "textbf", "\\dd"):
            assert macro not in text, f"{slug}: {macro} survived into the metadata field"
        left = set(re.findall(r"\\[a-zA-Z]+", text)) - ap.MATH_MACROS_ARXIV_RENDERS
        assert not left, f"{slug}: unrenderable macros survived: {sorted(left)}"


def test_plain_abstract_attributes_rather_than_dropping():
    """Dropping a citation leaves the abstract asserting someone else's result unattributed.

    Checked per paper, because each abstract cites a different set and a test pinned to one paper's
    citations would pass vacuously on the other two.
    """
    for slug in paper_sources.PAPERS:
        text = ap.plain_abstract(paper_sources.paper_dir(slug))
        source = paper_sources.paper_tex(slug)
        abstract_src = source[source.index(r"\begin{abstract}") : source.index(r"\end{abstract}")]
        for key in set(re.findall(r"\\cite[tp]\{([^}]*)\}", abstract_src)):
            for one in (k.strip() for k in key.split(",")):
                surname = ap._author_year(paper_sources.paper_dir(slug))[one].split()[0]
                assert surname in text, f"{slug}: {one} lost its attribution"
    # GEPA is the object of the series, so at least one abstract must cite it -- but not every one.
    # Paper II runs no optimizer at all, and an abstract of it citing GEPA would misdescribe the
    # experiments, so a blanket assertion here would be wrong rather than merely strict.
    citing_gepa = [
        slug
        for slug in paper_sources.PAPERS
        if "Agrawal" in ap.plain_abstract(paper_sources.paper_dir(slug))
    ]
    assert citing_gepa, "no abstract attributes GEPA, though the series is about it"
    assert "p2-repair-not-detection" not in citing_gepa, (
        "Paper II invokes no optimizer; citing GEPA in its abstract would misdescribe its design"
    )


def test_plain_abstract_fits_the_arxiv_field():
    """arXiv rejects an abstract over 1920 characters outright."""
    n = len(ap.plain_abstract())
    assert n <= 1920, f"abstract is {n} characters, over arXiv's 1920 limit"
    assert n > 800, f"abstract is {n} characters, which looks truncated rather than concise"


def test_author_year_is_read_from_the_bibliography():
    """Surnames come from references.bib. Deriving them from the cite key would invent people."""
    cites = ap._author_year(ap.PAPER)
    assert cites["agrawal2025gepa"].startswith("Agrawal")
    assert cites["lakens2018equivalence"].startswith("Lakens")
    # Single-author entries must not claim co-authors.
    single = [v for v in cites.values() if "et al." not in v]
    assert single, "no single-author entries resolved, so the 'et al.' branch is untested"


def test_the_author_block_has_no_placeholder():
    """The one blocker a script can check for itself.

    This paper carried "Author list TBD" for its whole life; a fixed checklist line reminding a
    human to fix it would now be a false statement standing in the output forever, which is the
    same stale-assertion failure the paper's own coverage guards exist to catch. So the reminder is
    derived from the file and disappears when it is satisfied.
    """
    assert ap.unresolved_placeholders() == [], (
        f"the author block still reads {ap.author_block()!r}, which arXiv cannot accept"
    )


def test_a_placeholder_author_block_is_detected():
    """The detector has to fire on the string it was built for, not merely pass on a clean file."""
    for bad in ("Author list TBD", "Anonymous Authors", "TODO fill in"):
        assert [w for w in ap._PLACEHOLDERS if w.lower() in bad.lower()], bad
    assert not [w for w in ap._PLACEHOLDERS if w.lower() in "Vladimir Rotariu".lower()]


def test_preflight_is_clean_and_checks_what_it_claims():
    for slug in paper_sources.PAPERS:
        problems = ap.preflight(paper_sources.paper_dir(slug))
        assert problems == [], f"{slug} should be submittable at HEAD: {problems}"
    src = inspect.getsource(ap.preflight)
    for concern in ("unresolved_placeholders", "pdfoutput_line", "ABSTRACT_LIMIT"):
        assert concern in src, f"preflight stopped checking {concern}"


def test_pdfoutput_is_declared_early_enough():
    """arXiv reads only the first five lines, and comments count toward that budget.

    Without the declaration, arXiv's pipeline may choose the DVI path, which cannot include a PDF
    figure --- so a paper with three PDF figures could be announced with all three missing.
    """
    line = ap.pdfoutput_line()
    assert line is not None, r"main.tex has no \pdfoutput=1"
    assert line <= 5, f"declared on line {line}, outside the window arXiv inspects"


def test_build_bbl_removes_the_old_file_before_regenerating():
    """Otherwise "exists" means "left over from a previous build", not "produced by this run".

    That was a real defect: the function ran bibtex, ignored its exit status, and then checked only
    that main.bbl existed --- which it always did. A bibtex failure would have shipped a stale
    bibliography while the docstring promised a fresh one.
    """
    src = inspect.getsource(ap.build_bbl)
    assert "unlink" in src, "a stale .bbl could be shipped"
    assert src.index("unlink") < src.index('"bibtex"'), "the deletion must precede the run"
    assert "main.blg" in src, "bibtex errors must be detected, not just a missing file"


def test_an_unsubmittable_paper_produces_no_package():
    """A tarball that cannot be submitted is worse than none: it looks like a finished artifact."""
    src = inspect.getsource(ap.main)
    # Matched on the call PREFIX, not the exact string. This read `preflight(paper)` and broke
    # when the call gained keyword arguments for `--anonymous` -- a guard failing on a change that
    # cannot affect what it guards. The ordering is the property; the argument list is not.
    # `.find` with an explicit assertion rather than `.index`, so a rename reports which name went
    # missing instead of raising a bare ValueError from inside the comparison.
    gate, build = src.find("preflight(paper"), src.find("build_bbl(paper")
    assert gate != -1, "main() no longer calls preflight(paper...); the build is ungated"
    assert build != -1, (
        "main() no longer calls build_bbl(paper...); this guard has lost its subject"
    )
    assert gate < build, "preflight must gate the build"
    assert "NOT SUBMITTABLE" in src


def test_an_unresolvable_citation_is_an_actionable_error(tmp_path):
    """Not a bare KeyError from a dict lookup two frames down."""
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(
        "\\begin{abstract}\nSee \\citep{ghost} for details.\n\\end{abstract}\n"
    )
    (paper / "references.bib").write_text("@misc{other,\n  title = {T},\n  year = {2020}\n}\n")
    with pytest.raises(RuntimeError, match="ghost"):
        ap.plain_abstract(paper)


def test_the_arxiv_limits_are_the_documented_ones():
    assert ap.ABSTRACT_LIMIT == 1920, "arXiv's documented abstract field limit"


def test_every_documented_page_count_is_found_rather_than_listed():
    """The page count is stated in three places, and all three have to be read.

    `tests/test_paper_docs_consistency.py` already ties README's table to SUBMISSION's, but both
    are hand-typed, so agreement between them says nothing about the built PDF -- the shared-error
    blind spot the papers' own register describes. The sites are swept rather than enumerated, and
    the sweep's own yield is asserted, because a guard that stops finding its subject reports
    nothing rather than failing.
    """
    sites = ap._page_sites()
    assert len(sites) == ap._EXPECTED_PAGE_SITES, (
        f"found {len(sites)} documented page-count sites, expected {ap._EXPECTED_PAGE_SITES}: "
        f"{[where for where, _ in sites]}"
    )
    for where, counts in sites:
        assert set(counts) == set(paper_sources.PAPERS), f"{where} does not state all three"
    for slug in paper_sources.PAPERS:
        assert ap.advertised_pages(slug) > 0


def test_a_page_count_that_disagrees_between_sites_is_an_error():
    """Not a silent pick of the first one found."""
    with pytest.raises(LookupError, match="disagree"):
        ap._reconcile_pages(
            "p1-conditional-lever",
            [("doc A", {"p1-conditional-lever": 39}), ("doc B", {"p1-conditional-lever": 38})],
        )


def test_a_page_count_that_drifted_from_the_build_blocks_the_package():
    for slug in paper_sources.PAPERS:
        n = ap.advertised_pages(slug)
        assert ap.stale_page_count(slug, n) is None
        message = ap.stale_page_count(slug, n + 1)
        assert message is not None
        assert str(n) in message and str(n + 1) in message


def test_the_build_refuses_a_stale_page_count_before_writing_the_tarball():
    """A package whose documented size is wrong is what someone quotes in a submission form."""
    src = inspect.getsource(ap.main)
    check, tar = src.find("stale_page_count("), src.find("tarfile.open")
    assert check != -1, "main() no longer compares the rehearsed page count against the docs"
    assert tar != -1, "this guard has lost its subject"
    assert check < tar, "the comparison must gate the tarball, not follow it"
    assert "return 1" in src[check:tar], "a stale page count must fail the build, not warn"
