#!/usr/bin/env python3
r"""Build a TMLR submission package for one paper of the series.

TMLR requires its own stylefile: "Submissions must be PDF files generated using the TMLR LaTeX
stylefile and template" (author guide, quoted 2026-09-10). These papers are `article` at 11pt
with `geometry[margin=1in]`, so three things have to move, and each fails quietly if it does not:

* `tmlr.sty` sets `\textwidth`, `\textheight` and `\topmargin` as raw dimensions. A `geometry`
  loaded afterwards recomputes all of them, so the paper would compile, look fine, and be laid
  out for the wrong journal.
* `tmlr.sty` expects a 10pt base.
* The bibliography has to be typeset with `tmlr.bst`, and the `main.bbl` sitting in the paper
  directory was produced by `plainnat` for the arXiv build. Shipping that one gives TMLR
  formatting from the wrong style, so the bbl is rebuilt inside the staged tree.

Everything else is reused from the arXiv packager rather than reimplemented: staging with the
`\input`s inlined, the anonymiser, the byte scan for identifying strings, and the compile
rehearsal. The two targets differ in style, not in what makes a package submittable.

ANONYMITY IS TWO MECHANISMS, NOT ONE. `\usepackage{tmlr}` without `[accepted]` renders
"Anonymous authors / Paper under double-blind review" in place of the author block, but it only
hides what the block contains. Measured on Paper I: the rendered PDF still printed the repository
URL once, from the companion-paper note in `shared/series.tex`, which no style option touches.
So `--anonymous` still runs the source anonymiser, and the byte scan still has to pass.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import make_arxiv_package as arxiv  # noqa: E402
import paper_sources  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
STYLE_DIR = ROOT / "docs/papers/tmlr"

#: Vendored rather than fetched at build time. Every other gate in this repo runs offline, and a
#: packaging step that reaches the network fails differently on a plane than it does in CI.
STYLE_FILES = ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")

_CLASS_FROM = r"\documentclass[11pt]{article}"
_CLASS_TO = "\\documentclass[10pt]{article}\n\\usepackage{tmlr}"
_GEOMETRY = r"\usepackage[margin=1in]{geometry}"
_GEOMETRY_TO = "% geometry dropped for TMLR: tmlr.sty sets the page dimensions as raw lengths"
_BIB_FROM = r"\bibliographystyle{plainnat}"
_BIB_TO = r"\bibliographystyle{tmlr}"


def tmlrise(tex: str) -> str:
    """Rewrite an inlined paper source to build against the TMLR stylefile.

    Each anchor is asserted to occur exactly once BEFORE anything is replaced. A style swap that
    silently matched nothing is the failure worth guarding: the document still compiles, still
    reads correctly, and is formatted for the venue you are not submitting to.
    """
    for anchor in (_CLASS_FROM, _GEOMETRY, _BIB_FROM):
        assert tex.count(anchor) == 1, (
            f"expected exactly one {anchor!r} to rewrite, found {tex.count(anchor)}; "
            "refusing to write a package styled for the wrong venue"
        )
    tex = tex.replace(_CLASS_FROM, _CLASS_TO)
    tex = tex.replace(_GEOMETRY, _GEOMETRY_TO)
    return tex.replace(_BIB_FROM, _BIB_TO)


def stage(
    dest: pathlib.Path,
    paper: pathlib.Path,
    *,
    anonymous: bool = False,
    anon_url: str | None = None,
) -> pathlib.Path:
    """Stage a flat TMLR tree: inlined source, figures, references and the three style files."""
    dest.mkdir(parents=True, exist_ok=True)
    tex = tmlrise(paper_sources.paper_tex(paper.name))
    if anonymous:
        tex = arxiv.anonymise_tex(tex, anon_url)
    (dest / "main.tex").write_text(tex, encoding="utf-8")

    shutil.copy2(paper / "references.bib", dest / "references.bib")
    if anonymous:
        bib = dest / "references.bib"
        bib.write_text(arxiv.anonymise_bib(bib.read_text(encoding="utf-8")), encoding="utf-8")

    for name in STYLE_FILES:
        shutil.copy2(STYLE_DIR / name, dest / name)
    arxiv.stage_figures(paper, dest, tex)
    return dest


def build_bbl_in_place(staged: pathlib.Path) -> pathlib.Path:
    """Produce `main.bbl` with `tmlr.bst`, inside the staged tree.

    Deliberately not `arxiv.build_bbl`, which runs in the paper directory and would leave a
    TMLR-styled bbl behind for the arXiv build to ship. The deletion first is the same trick that
    function uses: it turns "the file exists" into "this run produced it".
    """
    bbl = staged / "main.bbl"
    bbl.unlink(missing_ok=True)
    arxiv._run(["pdflatex", "-interaction=nonstopmode", "main.tex"], staged)
    bib = arxiv._run(["bibtex", "main"], staged)
    if not bbl.exists():
        raise RuntimeError(
            "bibtex produced no main.bbl under tmlr.bst, so the submission would print [?] "
            f"for every citation. bibtex said:\n{bib.stdout[-2000:]}"
        )
    return bbl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="dist")
    ap.add_argument("--paper", default=paper_sources.PAPERS[0], choices=paper_sources.PAPERS)
    ap.add_argument(
        "--anonymous",
        action="store_true",
        help="double-blind variant. TMLR is double-blind and the style package alone does not "
        "remove the repository URL in the companion-paper note",
    )
    ap.add_argument("--anon-url", default=None)
    args = ap.parse_args()

    paper = paper_sources.paper_dir(args.paper)
    print(
        f"packaging {paper_sources.PAPER_NUMERALS[args.paper]} for TMLR: "
        f"{paper_sources.PAPER_TITLES[args.paper]}"
        f"{'  [DOUBLE-BLIND]' if args.anonymous else ''}\n"
    )

    problems = arxiv.preflight(paper, anonymous=args.anonymous, anon_url=args.anon_url)
    if problems:
        print("NOT SUBMITTABLE, so no package was written:")
        for pr in problems:
            print(f"  - {pr}")
        return 1

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(
            pathlib.Path(tmp) / "tmlr", paper, anonymous=args.anonymous, anon_url=args.anon_url
        )
        build_bbl_in_place(staged)
        if args.anonymous:
            leaks = arxiv.identifying_leaks(staged)
            if leaks:
                print("STILL IDENTIFYING, so no package was written:")
                for where, token in leaks:
                    print(f"  - {where}: {token!r}")
                return 1
            print(f"OK  no identifying string survives ({len(arxiv.IDENTIFYING)} scanned)")

        result = arxiv.rehearse(staged)
        print(f"OK  rehearsed: {result}")
        prefix = "tmlr-anon" if args.anonymous else "tmlr"
        # Keep the compiled PDF. The staged tree is a TemporaryDirectory, so without this the
        # only anonymous TMLR PDF that has ever existed is deleted three lines later -- and it
        # is the file the submission uploads, and the file the OTHER two papers need in their
        # supplementary bundle (a reviewer of Paper I cannot otherwise check twelve numeric
        # claims it attributes to its companions).
        pdf = staged / "main.pdf"
        if pdf.exists():
            shutil.copy2(pdf, out / f"{prefix}-{args.paper}.pdf")
            print(f"OK  wrote {out / f'{prefix}-{args.paper}.pdf'}")
        tarball = out / f"{prefix}-{args.paper}.tar.gz"
        shutil.make_archive(str(tarball).removesuffix(".tar.gz"), "gztar", staged)
        print(f"OK  wrote {tarball}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
