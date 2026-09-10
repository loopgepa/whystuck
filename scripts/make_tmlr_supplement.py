#!/usr/bin/env python3
r"""Build the supplementary ZIP that ships beside one paper's TMLR submission.

TMLR takes "up to 100MB of supplementary material, such data, source code or illustrative
videos; all supplementary materials must be in PDF or ZIP format", and "like submissions,
supplementary material must be anonymized" (author guide, quoted 2026-09-10). That removes the
third-party mirror-hosting step entirely: the artifact travels with the paper.

Two things go in the bundle.

**The artifact.** `make_anon_mirror.py` writes a de-identified snapshot tree. Nothing zipped it:
the 6.4MB figure quoted in `SUBMISSION.md` came from a zip made by hand once, which no script
reproduced and which was six days stale by the time anyone looked.

**Both companion papers.** A reviewer of Paper I cannot read Papers II and III, and Paper I
attributes numeric claims to both, so those claims are unverifiable to the one person who has to
judge them. COPE, on divided publication, says authors "should cross reference or include a copy
of the companion paper"; NeurIPS says to "include a copy of the cited anonymized submission in
the supplementary material". The anonymous TMLR PDFs already exist, so this costs nothing.

The bundle is scanned twice before it is written: `identifying_leaks` over the text sources, and
`pdf_stream_leaks` over every PDF, because the first cannot see inside a compressed PDF stream.
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
TMLR_LIMIT_BYTES = 100 * 1024 * 1024


def companion_pdfs(slug: str, dist: pathlib.Path) -> list[pathlib.Path]:
    """The anonymous TMLR PDFs of the OTHER two papers, in series order."""
    out = []
    for other in paper_sources.PAPERS:
        if other == slug:
            continue
        pdf = dist / f"tmlr-anon-{other}.pdf"
        if not pdf.is_file():
            raise FileNotFoundError(
                f"{pdf} is missing. Build it first:\n"
                f"  python scripts/make_tmlr_package.py --paper {other} --anonymous "
                f'--anon-url "the supplementary material accompanying this submission"'
            )
        out.append(pdf)
    return out


def stage(dest: pathlib.Path, slug: str, mirror: pathlib.Path, dist: pathlib.Path) -> pathlib.Path:
    dest.mkdir(parents=True, exist_ok=True)
    if not mirror.is_dir():
        raise FileNotFoundError(
            f"{mirror} is missing. Build it first:\n"
            "  python scripts/make_anon_mirror.py --dest dist/anon-mirror"
        )
    shutil.copytree(mirror, dest / "artifact", symlinks=False)
    comp = dest / "companion-papers"
    comp.mkdir()
    for pdf in companion_pdfs(slug, dist):
        other = pdf.stem.removeprefix("tmlr-anon-")
        shutil.copy2(pdf, comp / f"paper-{paper_sources.PAPER_NUMERALS[other]}.pdf")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", default=paper_sources.PAPERS[0], choices=paper_sources.PAPERS)
    ap.add_argument("--outdir", default="dist")
    ap.add_argument("--mirror", default="dist/anon-mirror")
    args = ap.parse_args()

    dist = pathlib.Path(args.outdir)
    dist.mkdir(parents=True, exist_ok=True)
    numeral = paper_sources.PAPER_NUMERALS[args.paper]
    print(f"supplementary bundle for Paper {numeral} ({args.paper})\n")

    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(pathlib.Path(tmp) / "supp", args.paper, pathlib.Path(args.mirror), dist)

        leaks = arxiv.identifying_leaks(staged)
        if leaks:
            print("STILL IDENTIFYING in the sources, so no bundle was written:")
            for where, token in leaks:
                print(f"  - {where}: {token!r}")
            return 1
        print(f"OK  no identifying string in the staged sources ({len(arxiv.IDENTIFYING)} scanned)")

        pdfs = sorted(staged.rglob("*.pdf"))
        pdf_leaks = arxiv.pdf_stream_leaks(pdfs)
        if pdf_leaks:
            print("STILL IDENTIFYING inside a PDF, so no bundle was written:")
            for where, token in pdf_leaks:
                print(f"  - {where}: {token!r}")
            return 1
        print(f"OK  no identifying string inside any of {len(pdfs)} PDFs (streams inflated)")

        base = dist / f"tmlr-supplement-{args.paper}"
        shutil.make_archive(str(base), "zip", staged)
        size = base.with_suffix(".zip").stat().st_size
        pct = 100 * size / TMLR_LIMIT_BYTES
        print(
            f"OK  wrote {base.with_suffix('.zip')} "
            f"({size / 1048576:.1f} MB, {pct:.1f}% of TMLR's 100MB limit)"
        )
        if size > TMLR_LIMIT_BYTES:
            print("OVER THE LIMIT: TMLR will not accept this bundle")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
