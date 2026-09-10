"""The TMLR build: what the style swap must change, and what it must refuse.

TMLR requires its own stylefile ("Submissions must be PDF files generated using the TMLR LaTeX
stylefile and template", author guide, quoted 2026-09-10), and these papers are `article` at 11pt
with `geometry[margin=1in]`. Three things collide and each is silent if it goes wrong:
`tmlr.sty` sets `\\textwidth`/`\\topmargin` as raw dimensions that a later `geometry` would
overwrite, it `\\RequirePackage`s natbib, and it expects a 10pt base.

Every transformation below asserts its anchor before editing, because a style swap that matched
nothing would produce a document that still compiles, still looks reasonable and is formatted for
the wrong venue.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import make_tmlr_package as tm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_vendored_style_files_are_present() -> None:
    """Vendored rather than downloaded: the build has to work offline like every other gate."""
    for name in tm.STYLE_FILES:
        path = tm.STYLE_DIR / name
        assert path.is_file(), f"{name} is not vendored in {tm.STYLE_DIR}"
        assert path.stat().st_size > 0, f"{name} is empty"


def test_the_style_swap_moves_to_ten_point_and_loads_tmlr() -> None:
    tex = tm.tmlrise(
        r"\documentclass[11pt]{article}"
        "\n"
        r"\usepackage[margin=1in]{geometry}"
        "\n"
        r"\bibliographystyle{plainnat}"
    )
    assert r"\documentclass[10pt]{article}" in tex
    assert r"\usepackage{tmlr}" in tex
    assert "11pt" not in tex


def test_the_style_swap_drops_geometry_because_tmlr_sets_the_page_itself() -> None:
    tex = tm.tmlrise(
        r"\documentclass[11pt]{article}"
        "\n"
        r"\usepackage[margin=1in]{geometry}"
        "\n"
        r"\bibliographystyle{plainnat}"
    )
    assert "geometry" not in tex.replace("% geometry", ""), (
        "geometry survived the swap, so it would overwrite tmlr.sty's page dimensions"
    )


def test_the_style_swap_moves_the_bibliography_style() -> None:
    tex = tm.tmlrise(
        r"\documentclass[11pt]{article}"
        "\n"
        r"\usepackage[margin=1in]{geometry}"
        "\n"
        r"\bibliographystyle{plainnat}"
    )
    assert r"\bibliographystyle{tmlr}" in tex
    assert "plainnat" not in tex


@pytest.mark.parametrize(
    "missing",
    [
        r"\usepackage[margin=1in]{geometry}" "\n" r"\bibliographystyle{plainnat}",
        r"\documentclass[11pt]{article}" "\n" r"\bibliographystyle{plainnat}",
        r"\documentclass[11pt]{article}" "\n" r"\usepackage[margin=1in]{geometry}",
    ],
)
def test_a_missing_anchor_refuses_rather_than_silently_doing_nothing(missing: str) -> None:
    """A swap that matched nothing formats the paper for the wrong venue and says it worked."""
    with pytest.raises(AssertionError):
        tm.tmlrise(missing)


def test_every_paper_carries_all_three_anchors_the_swap_needs() -> None:
    """The guard above only bites if the real sources still look like what it expects."""
    for slug in tm.paper_sources.PAPERS:
        tex = tm.paper_sources.paper_tex(slug)
        assert tex.count(r"\documentclass[11pt]{article}") == 1, slug
        assert tex.count(r"\usepackage[margin=1in]{geometry}") == 1, slug
        assert tex.count(r"\bibliographystyle{plainnat}") == 1, slug
