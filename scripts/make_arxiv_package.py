"""Assemble and verify an arXiv upload, including the one file it is easiest to forget.

WHY THIS EXISTS. arXiv does not run BibTeX. It runs pdflatex a few times over whatever you upload,
so a submission without `main.bbl` compiles and posts with every citation rendered as a bare
`[?]` --- a silent, public, and permanent-looking failure. `main.bbl` is a build artifact and is
correctly gitignored in this repository, which is exactly what makes it easy to omit from the
tarball. So the packaging step regenerates it and then REHEARSES the arXiv build in a clean
directory with BibTeX deliberately not run, failing if any citation or reference is undefined.

It also prints a plain-text abstract for the submission form. The abstract in the PDF carries
`\\citep`, `\\emph` and `\\ref`, none of which mean anything in arXiv's metadata field.

Usage: `python scripts/make_arxiv_package.py [--outdir DIR]`. Offline, $0.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_sources  # noqa: E402

#: Which paper to package. The series has three (`docs/papers/README.md`); each is submitted
#: separately, so this is a required choice rather than a default worth guessing at. `--paper`
#: overrides it.
PAPER = paper_sources.paper_dir(paper_sources.PAPERS[0])
#: Everything arXiv needs and nothing it does not. `.bbl` is added separately because it is a build
#: artifact rather than a tracked source file, which is the whole point of this script.
SOURCES = ("references.bib",)
FIGURES = "figures"

#: Standard TeX math macros arXiv's abstract field renders, so they may survive into the plain
#: abstract. Everything else must be converted to words, because in the metadata field it resolves
#: against nothing: a document-local macro like `\dd` renders as empty, and a `\ref` renders as a
#: dangling "the paper". The allowlist is deliberately short -- if an abstract needs a macro not
#: here, the right move is usually to reword the abstract, not to extend this set.
MATH_MACROS_ARXIV_RENDERS = frozenset(
    {
        r"\sigma",
        r"\Delta",
        r"\approx",
        r"\times",
        r"\le",
        r"\ge",
        r"\pm",
        r"\to",
    }
)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def build_bbl(paper: Path = PAPER) -> Path:
    """Regenerate `main.bbl` from `references.bib`, so the tarball cannot ship a stale one.

    The deletion is the load-bearing line. An earlier version of this function ran bibtex and then
    merely checked that `main.bbl` existed --- which it always did, left over from a previous build
    --- so a bibtex failure would have shipped a stale bibliography while the docstring claimed
    otherwise. Removing the file first turns "exists" into "was produced by this run".
    """
    bbl = paper / "main.bbl"
    bbl.unlink(missing_ok=True)
    _run(["pdflatex", "-interaction=nonstopmode", "main.tex"], paper)
    bib = _run(["bibtex", "main"], paper)
    _run(["pdflatex", "-interaction=nonstopmode", "main.tex"], paper)
    if not bbl.exists():
        raise RuntimeError(
            "bibtex produced no main.bbl, so the upload would post with [?] citations. "
            f"bibtex said:\n{bib.stdout[-2000:]}"
        )
    blg = (paper / "main.blg").read_text(errors="ignore") if (paper / "main.blg").exists() else ""
    fatal = [ln for ln in blg.splitlines() if ln.startswith("I couldn't") or "---line" in ln]
    if fatal:
        raise RuntimeError(
            "bibtex reported errors, so main.bbl is unreliable:\n" + "\n".join(fatal)
        )
    return bbl


def stage(dest: Path, paper: Path = PAPER) -> Path:
    """Copy what arXiv needs into a flat tree, with every `\\input` inlined.

    The inlining is load-bearing since the split. Each paper `\\input`s three files from
    `../shared/` -- the preamble, the harness section and the companion-paper note -- and arXiv
    receives a flat directory with no parent to resolve them against. Copying `main.tex` verbatim
    would upload a file whose first `\\input` fails, which pdflatex survives in
    `nonstopmode`: the submission would post with no packages, no title and no harness section.
    `rehearse()` would catch it, but inlining means there is nothing to catch.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "main.tex").write_text(paper_sources.paper_tex(paper.name), encoding="utf-8")
    for name in (*SOURCES, "main.bbl"):
        shutil.copy2(paper / name, dest / name)
    shutil.copytree(paper / FIGURES, dest / FIGURES, dirs_exist_ok=True)
    return dest


def rehearse(staged: Path) -> dict[str, object]:
    """Compile the staged tree the way arXiv will: pdflatex only, never bibtex."""
    for _ in range(3):
        _run(["pdflatex", "-interaction=nonstopmode", "main.tex"], staged)
    log = (staged / "main.log").read_text(errors="ignore")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    return {
        "pages": int(pages.group(1)) if pages else 0,
        "undefined_citations": re.findall(r"Citation .*? undefined", log),
        "undefined_references": re.findall(r"Reference .*? undefined", log),
        "errors": [ln for ln in log.splitlines() if ln.startswith("!")],
        "missing_files": re.findall(r"File .*? not found", log),
    }


def plain_abstract(paper: Path = PAPER) -> str:
    """The abstract as arXiv's metadata field wants it: no macros, no cross-references.

    Simple `$...$` math is left alone --- arXiv renders it --- but everything that resolves against
    the document (citations, `\\ref`) has to become words, because in the metadata field it resolves
    against nothing.
    """
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if m is None:
        raise RuntimeError("no abstract found in main.tex")
    ab = m.group(1).strip()
    # Our own macros first. `\dd` is defined in shared/preamble.tex, so arXiv's metadata field --
    # which sees the abstract as text and has no preamble -- would render it as nothing. Standard
    # math macros are a different case and are allowlisted below rather than expanded.
    ab = re.sub(r"\\dd\b", r"\\Delta", ab)
    # Citations become author-year, read out of references.bib rather than guessed from the key:
    # inventing a surname to fill an abstract would be a fabrication, and dropping the citation
    # entirely leaves the abstract asserting someone else's result unattributed.
    cites = _author_year(paper)

    def _render(mm: re.Match[str], fmt: str) -> str:
        names = []
        for key in (k.strip() for k in mm.group(1).split(",")):
            if key not in cites:
                raise RuntimeError(
                    f"cited key {key!r} has no author/year in references.bib, so it cannot be "
                    "rendered for the metadata field; fix the entry rather than dropping the cite"
                )
            names.append(cites[key])
        return fmt.format(", ".join(names))

    for macro, fmt in ((r"\\citep", "({})"), (r"\\citet", "{}")):
        ab = re.sub(macro + r"\{([^}]*)\}", lambda mm, f=fmt: _render(mm, f), ab)
    ab = re.sub(r"\s+\(", " (", ab)
    ab = re.sub(r"\\emph\{([^}]*)\}", r"\1", ab)
    ab = re.sub(r"\\text(?:bf|it|tt)\{([^}]*)\}", r"\1", ab)
    ab = re.sub(r"Appendix~\\ref\{[^}]*\}", "an appendix", ab)
    ab = re.sub(r"\S*~?\\ref\{[^}]*\}", "the paper", ab)
    # LaTeX dash ligatures have no meaning in a plain-text field. Convert to the characters they
    # stand for rather than leaving "--" for a reader to decode.
    ab = ab.replace("~", " ").replace("---", "\u2014").replace("--", "\u2013")
    ab = re.sub(r"[ \t]+", " ", ab).strip()
    leftover = sorted(set(re.findall(r"\\[a-zA-Z]+", ab)) - MATH_MACROS_ARXIV_RENDERS)
    if leftover:
        raise RuntimeError(f"unconverted LaTeX left in the plain abstract: {leftover}")
    return ab


#: Words that mean "we have not decided yet". A paper carrying any of these in its author block
#: cannot be submitted, and the check is dynamic rather than a fixed checklist line so that the
#: reminder disappears when it is satisfied instead of standing there falsely.
_PLACEHOLDERS = ("TBD", "TODO", "Anonymous", "XXX", "FIXME", "Author list")


def author_block(paper: Path = PAPER) -> str:
    m = re.search(r"\\author\{(.*?)\}\s*\n\\date", (paper / "main.tex").read_text(), re.DOTALL)
    if m is None:
        raise RuntimeError("no author block found in main.tex")
    return " ".join(m.group(1).split())


def unresolved_placeholders(paper: Path = PAPER) -> list[str]:
    block = author_block(paper)
    return [w for w in _PLACEHOLDERS if w.lower() in block.lower()]


def _author_year(paper: Path) -> dict[str, str]:
    """key -> "Surname et al., Year", parsed from references.bib."""
    bib = (paper / "references.bib").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for entry in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.DOTALL):
        key, body = entry.group(1).strip(), entry.group(2)
        author = re.search(r"author\s*=\s*\{(.*?)\}(?:,|\s*\n)", body, re.DOTALL)
        year = re.search(r"year\s*=\s*\{?(\d{4})", body)
        if not (author and year):
            continue
        first = author.group(1).split(" and ")[0].strip().strip("{}")
        surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
        many = " and " in author.group(1)
        out[key] = f"{surname}{' et al.' if many else ''}, {year.group(1)}"
    return out


#: arXiv's hard limit on the abstract metadata field, in characters. Exceeding it is a rejection at
#: the form, not a warning.
ABSTRACT_LIMIT = 1920


def pdfoutput_line(paper: Path = PAPER) -> int | None:
    """Which line declares `\\pdfoutput=1`, or None. Must be within the first five."""
    for i, line in enumerate(
        (paper / "main.tex").read_text(encoding="utf-8").splitlines()[:5], start=1
    ):
        if line.strip().startswith(r"\pdfoutput=1"):
            return i
    return None


def preflight(paper: Path = PAPER) -> list[str]:
    """Everything the script can decide for itself. Returns blocking problems, empty if clean.

    Checked BEFORE anything is written, because a tarball that cannot be submitted is worse than no
    tarball: it looks like a finished artifact.
    """
    problems = []
    stale = unresolved_placeholders(paper)
    if stale:
        problems.append(f"the author block still contains {stale}: {author_block(paper)!r}")
    if pdfoutput_line(paper) is None:
        problems.append(
            r"no \pdfoutput=1 in the first five lines of main.tex. This submission has PDF "
            "figures, and a DVI run cannot include them, so they could be announced missing"
        )
    n = len(plain_abstract(paper))
    if n > ABSTRACT_LIMIT:
        problems.append(f"the plain abstract is {n} characters, over arXiv's {ABSTRACT_LIMIT}")
    return problems


#: The two choices arXiv forces at the form, with the reasoning, because both are consequential and
#: one of them is irreversible.
LICENCE_NOTE = """  Licence:  CC BY 4.0. DECIDED. Select it on the form; do not take the default.
      The paper's contribution is a reporting practice plus a released artifact (every cassette,
      every pre-registration, the audit) and it asks other people to re-run and re-analyse it.
      arXiv's Perpetual Non-Exclusive licence, which is the default, grants arXiv distribution
      rights while limiting re-use by everyone else, and that works against the point. The
      conclusion of the paper now states the same thing, so the metadata and the text agree.
      Code in this repository stays Apache-2.0; the two are separate grants.
      IRREVOCABLE: arXiv cannot change a licence after announcement. Only a later version can
      carry a different one, and the version posted stays licensed as posted."""

CATEGORY_NOTE = """  Categories:  primary cs.LG, cross-list cs.CL.
      The object of study is an optimizer's behaviour and how to measure it, which is cs.LG;
      the tasks are instruction-following and code, which is why cs.CL is the cross-list.
      DECIDE THIS BEFORE HUNTING AN ENDORSER: endorsement is granted per endorsement domain,
      so the primary category determines which endorsement you need."""

ENDORSEMENT_NOTE = """  Endorsement:  as an independent researcher, expect to need one, and expect
      to ask a person for it. arXiv tightened this on 21 January 2026: an institutional email
      address is no longer sufficient on its own. The automatic route now needs an institutional
      address AND prior authorship in the same endorsement domain; with neither, the only route
      is a personal endorsement from an established arXiv author in that domain (an endorser
      needs roughly five submissions there). arXiv staff cannot endorse and cannot waive it.
      What to send: the built PDF, one paragraph on what the paper measures, and the repository
      link. The paper's own strongest hook is that it invalidated its own headline twice and
      releases the cassettes to prove it."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dist", help="where to write the tarball")
    ap.add_argument(
        "--paper",
        default=paper_sources.PAPERS[0],
        choices=paper_sources.PAPERS,
        help="which paper of the series to package (each is submitted separately)",
    )
    args = ap.parse_args()
    paper = paper_sources.paper_dir(args.paper)
    print(
        f"packaging {paper_sources.PAPER_NUMERALS[args.paper]}: "
        f"{paper_sources.PAPER_TITLES[args.paper]}\n"
    )

    problems = preflight(paper)
    if problems:
        print("NOT SUBMITTABLE, so no package was written:")
        for pr in problems:
            print(f"  - {pr}")
        return 1

    build_bbl(paper)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp) / "arxiv", paper)
        result = rehearse(staged)
        bad = (
            result["undefined_citations"]
            or result["undefined_references"]
            or result["errors"]
            or result["missing_files"]
        )
        if bad or not result["pages"]:
            print("ARXIV REHEARSAL FAILED (this is what the posted paper would look like):")
            for k in ("undefined_citations", "undefined_references", "errors", "missing_files"):
                if result[k]:
                    print(f"  {k}: {result[k]}")
            return 1
        tar = out / f"arxiv-{args.paper}.tar.gz"
        with tarfile.open(tar, "w:gz") as t:
            for item in sorted(staged.iterdir()):
                if item.name.startswith("main.") and item.suffix not in (".tex", ".bbl"):
                    continue  # aux/log/pdf are byproducts of the rehearsal, not inputs
                t.add(item, arcname=item.name)
        size_kb = tar.stat().st_size / 1024
        print(f"OK  {tar}  ({size_kb:.0f} kB)")
        print(
            f"OK  rehearsed with pdflatex only, no bibtex: {result['pages']} pages, "
            "0 undefined citations"
        )
        with tarfile.open(tar) as t:
            print(
                "    contents: " + ", ".join(sorted(m.name for m in t.getmembers() if m.isfile()))
            )

    print("\n--- plain-text abstract for the submission form ---")
    print(plain_abstract(paper))
    print(f"\n--- author block ---\n    {author_block(paper)}")
    print(f"    abstract: {len(plain_abstract(paper))} characters of {ABSTRACT_LIMIT}")
    print(f"    \\pdfoutput=1 on line {pdfoutput_line(paper)} (must be <= 5)")
    print("\n--- decisions the form will ask for ---")
    for note in (LICENCE_NOTE, CATEGORY_NOTE, ENDORSEMENT_NOTE):
        print(note + "\n")
    print("--- the one thing this script cannot check ---")
    print(
        "  [ ] the repository must be public. The abstract claims a release and the title block\n"
        "      prints the URL under the author's name, so posting while it is private ships a\n"
        "      paper citing a 404."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
