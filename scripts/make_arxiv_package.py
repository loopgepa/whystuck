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
import tarfile
import tempfile
from pathlib import Path

PAPER = Path("docs/paper")
#: Everything arXiv needs and nothing it does not. `.bbl` is added separately because it is a build
#: artifact rather than a tracked source file, which is the whole point of this script.
SOURCES = ("main.tex", "references.bib")
FIGURES = "figures"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def build_bbl(paper: Path = PAPER) -> Path:
    """Regenerate `main.bbl` from `references.bib`, so the tarball cannot ship a stale one."""
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        _run(cmd, paper)
    bbl = paper / "main.bbl"
    if not bbl.exists():
        raise RuntimeError("bibtex produced no main.bbl; the upload would post with [?] citations")
    return bbl


def stage(dest: Path, paper: Path = PAPER) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
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
    # Citations become author-year, read out of references.bib rather than guessed from the key:
    # inventing a surname to fill an abstract would be a fabrication, and dropping the citation
    # entirely leaves the abstract asserting someone else's result unattributed.
    cites = _author_year(paper)
    for macro, fmt in ((r"\\citep", "({})"), (r"\\citet", "{}")):
        for mm in re.finditer(macro + r"\{([^}]*)\}", ab):
            names = ", ".join(cites[k.strip()] for k in mm.group(1).split(","))
            ab = ab.replace(mm.group(0), fmt.format(names))
    ab = re.sub(r"\s+\(", " (", ab)
    ab = re.sub(r"\\emph\{([^}]*)\}", r"\1", ab)
    ab = re.sub(r"\\text(?:bf|it|tt)\{([^}]*)\}", r"\1", ab)
    ab = re.sub(r"Appendix~\\ref\{[^}]*\}", "an appendix", ab)
    ab = re.sub(r"\S*~?\\ref\{[^}]*\}", "the paper", ab)
    # LaTeX dash ligatures have no meaning in a plain-text field. Convert to the characters they
    # stand for rather than leaving "--" for a reader to decode.
    ab = ab.replace("~", " ").replace("---", "\u2014").replace("--", "\u2013")
    ab = re.sub(r"[ \t]+", " ", ab).strip()
    if "\\" in ab:
        leftover = sorted(set(re.findall(r"\\[a-zA-Z]+", ab)))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dist", help="where to write the tarball")
    args = ap.parse_args()

    build_bbl()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp) / "arxiv")
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
        tar = out / "arxiv-submission.tar.gz"
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
    print(plain_abstract())
    print(f"\n--- author block ---\n    {author_block()}")
    stale = unresolved_placeholders()
    print("\n--- still required from a human ---")
    items = [
        "the repository must be public, or the abstract's release claim and the URL in the "
        "author block both point at nothing",
        "a licence (arXiv default vs CC BY) and a primary category (cs.LG or cs.CL, cross-list "
        "the other)",
        "endorsement, if this is a first submission to that archive",
    ]
    if stale:
        items.insert(0, f"a real author list: the block still contains {stale}")
    for item in items:
        print(f"  [ ] {item}")
    return 1 if stale else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
