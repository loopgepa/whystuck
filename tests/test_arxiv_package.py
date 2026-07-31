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


def test_the_rehearsal_fails_the_build_rather_than_warning():
    """Undefined citations must be a non-zero exit, not a line of output someone skims past."""
    src = inspect.getsource(ap.main)
    assert "undefined_citations" in src and "return 1" in src


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
    assert src.index("preflight(paper)") < src.index("build_bbl(paper)"), (
        "preflight must gate the build"
    )
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
