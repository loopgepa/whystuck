"""Guards on the double-blind build.

WHY THIS EXISTS. The venue survey of 2026-08-12 found that TMLR is the only venue whose length
policy and acceptance criteria fit these papers -- and that TMLR, like ICLR and NeurIPS, is
double-blind, while all three papers print the author's name, affiliation and the repository URL in
their title block and name the repository again in the abstract. The submission is de-anonymised by
its own reproducibility apparatus.

So `--anonymous` exists. And an anonymiser is the single worst place in this repository for a
silent no-op: it produces a file that LOOKS anonymised, gets uploaded, and cannot be recalled. That
is this series' own subject -- a check reporting success about something it never did -- with a
consequence that is public and permanent.

The load-bearing test here is therefore not "the anonymous build has no leaks" but
`test_the_leak_scan_is_not_vacuous`: the same scan, run over the UNMODIFIED sources, must find
them. A scanner that finds nothing anywhere would pass every other test in this file.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import make_arxiv_package as pkg  # noqa: E402
import paper_sources  # noqa: E402


def _tree(tmp_path: pathlib.Path, **files: str) -> pathlib.Path:
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


def test_the_leak_scan_is_not_vacuous(tmp_path: pathlib.Path) -> None:
    """Run the scan over the real, UNMODIFIED sources. It must find the author.

    Without this, a scanner broken in any way -- an empty token list, a bad encode, a rglob that
    matches nothing -- makes every other assertion in this file pass while the anonymiser ships
    the author's name.
    """
    staged = _tree(
        tmp_path,
        **{
            "main.tex": paper_sources.paper_tex(paper_sources.PAPERS[0]),
            "references.bib": (
                pathlib.Path("docs/papers/shared/references.bib").read_text(encoding="utf-8")
            ),
        },
    )
    leaks = pkg.identifying_leaks(staged)
    tokens = {t for _f, t in leaks}
    assert "Rotariu" in tokens, f"the scan missed the author's name in real sources: {leaks}"
    assert "loopgepa" in tokens, "the scan missed the repository owner"
    assert {f for f, _t in leaks} >= {"main.tex", "references.bib"}, (
        f"the scan did not look at both staged files: {sorted({f for f, _t in leaks})}"
    )


@pytest.mark.parametrize("slug", paper_sources.PAPERS)
def test_every_paper_anonymises_to_nothing_identifying(slug: str, tmp_path: pathlib.Path) -> None:
    staged = _tree(
        tmp_path,
        **{
            "main.tex": pkg.anonymise_tex(paper_sources.paper_tex(slug), "example.invalid/anon"),
            "references.bib": pkg.anonymise_bib(
                pathlib.Path("docs/papers/shared/references.bib").read_text(encoding="utf-8")
            ),
        },
    )
    assert pkg.identifying_leaks(staged) == []


@pytest.mark.parametrize("slug", paper_sources.PAPERS)
def test_anonymising_keeps_the_release_claim_pointing_somewhere(slug: str) -> None:
    """The abstract claims an artifact release. Deleting the URL would leave it citing nothing."""
    out = pkg.anonymise_tex(paper_sources.paper_tex(slug), "example.invalid/anon")
    assert "example.invalid/anon" in out
    assert pkg.ANONYMOUS_AUTHOR in out


def test_a_missing_author_block_raises_rather_than_passing_through() -> None:
    """A reworded title block must fail loudly. This is the silent-no-op case.

    `re.subn` returning 0 is not an error in Python, so without the count check the function would
    return the input unchanged and every downstream step would report success.
    """
    with pytest.raises(RuntimeError, match="author block"):
        pkg.anonymise_tex(r"\title{x}" "\n" r"\date{now}" "\n", "u")


def test_a_missing_repository_url_raises() -> None:
    with pytest.raises(RuntimeError, match="repository URL"):
        pkg.anonymise_tex(r"\author{A}" "\n" r"\date{now}" "\n", "u")


def test_the_bib_comment_naming_the_project_is_stripped() -> None:
    """It renders as nothing and ships in the tarball anyway, which is why the scan reads bytes."""
    bib = "% Bibliography for the WhyStuck GEPA validation study.\n@misc{a, year={2026}}\n"
    assert "whystuck" not in pkg.anonymise_bib(bib).lower()
    assert "@misc{a" in pkg.anonymise_bib(bib), "stripping the comment removed an entry"


def test_placeholders_stay_forbidden_in_anonymous_mode(monkeypatch) -> None:
    """`Anonymous` leaves the placeholder list under --anonymous; the rest must not.

    Dropping the whole check there would make the double-blind build the one build that can ship
    `TODO` as its author.
    """
    assert "Anonymous" not in pkg._PLACEHOLDERS_EVEN_WHEN_ANONYMOUS
    for token in ("TBD", "TODO", "XXX", "FIXME"):
        assert token in pkg._PLACEHOLDERS_EVEN_WHEN_ANONYMOUS
    monkeypatch.setattr(pkg, "ANONYMOUS_AUTHOR", "Anonymous authors TODO affiliation")
    assert pkg.unresolved_placeholders(anonymous=True) == ["TODO"]


def test_preflight_refuses_anonymous_without_a_url() -> None:
    problems = pkg.preflight(anonymous=True, anon_url=None)
    assert any("anon-url" in p for p in problems), problems
    assert pkg.preflight(anonymous=True, anon_url="example.invalid/x") == []
