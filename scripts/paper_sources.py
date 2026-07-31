#!/usr/bin/env python
"""The single source of truth for where the three papers live and how to read them.

One 46-page draft was split into three papers (see `docs/papers/README.md`). Every claim checker
and coverage guard in this repository used to open one hard-coded path, `docs/paper/main.tex`. If
each of them had grown its own list of three paths, the split would have introduced exactly the
class of defect the study is about: a check that quietly stops covering something. So all of them
read through here instead.

Two views, and the distinction matters:

`all_tex()` concatenates the three papers with their shared inputs expanded. This is the view the
claim checker and the coverage guards use, and it is the right one for the question they ask --
"does the paper's prose agree with the artifact" is a question about the series, not about which
file a sentence landed in. It also means the ~155 pinned-phrase checks written against the
monolith keep working unchanged: a sentence that survived the split into *some* paper is still
found, and one that was dropped yields NaN and fails loudly.

`paper_tex(slug)` reads one paper. This is the view the per-paper checks use, for the questions
concatenation cannot answer: does each paper still contain its own headline numbers, and does a
number that appears in two papers agree between them.

Shared `\\input` files are expanded rather than read separately, because a number in
`shared/loop.tex` is a number the reader of each paper sees, and a guard that missed it would be
under-counting coverage. Expansion is deliberately shallow-recursive and refuses to loop.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

#: The three papers, in series order. Slugs are directory names under `docs/papers/`.
#: Order is load-bearing for `all_tex()`, whose output several guards hash and diff.
PAPERS: tuple[str, ...] = (
    "p1-conditional-lever",
    "p2-repair-not-detection",
    "p3-silent-no-op",
)

#: Human-facing roman numeral per paper, as the papers' own cross-references use ("Paper~II").
PAPER_NUMERALS: dict[str, str] = {
    "p1-conditional-lever": "I",
    "p2-repair-not-detection": "II",
    "p3-silent-no-op": "III",
}

#: One-line description, used by the docs generator and the navigation README.
PAPER_TITLES: dict[str, str] = {
    "p1-conditional-lever": (
        "Whether Optimizing the Loop Beats Optimizing the Prompt Depends on One Number"
    ),
    "p2-repair-not-detection": "What a Checker Cannot Buy Is the Repair",
    "p3-silent-no-op": "Your Loop Optimizer Is Not Optimizing Your Loop",
}

PAPERS_DIR = Path("docs/papers")
SHARED_DIR = PAPERS_DIR / "shared"

#: `\input{...}` with an optional `.tex`. The papers only ever input from `../shared/`, and the
#: pattern is kept tight so a stray `\input` elsewhere fails visibly rather than being resolved
#: against an unexpected directory.
_INPUT = re.compile(r"\\input\{([^}]+)\}")

_MAX_INPUT_DEPTH = 4


def paper_dir(slug: str) -> Path:
    if slug not in PAPERS:
        raise KeyError(f"unknown paper slug {slug!r}; known: {', '.join(PAPERS)}")
    return PAPERS_DIR / slug


def paper_path(slug: str) -> Path:
    return paper_dir(slug) / "main.tex"


def _expand(text: str, base: Path, depth: int = 0) -> str:
    """Inline every `\\input{...}` relative to `base`.

    Raises on a missing input rather than leaving the directive in place: a paper whose shared
    section silently failed to resolve would still build a PDF (LaTeX would error, but a guard
    reading the source would not), and the guards are the thing we are protecting.
    """
    if depth > _MAX_INPUT_DEPTH:
        raise RecursionError(f"\\input nesting deeper than {_MAX_INPUT_DEPTH} under {base}")

    def sub(match: re.Match[str]) -> str:
        target = match.group(1)
        path = (base / target).resolve()
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        if not path.is_file():
            raise FileNotFoundError(f"{base}: \\input{{{target}}} resolves to missing {path}")
        return _expand(path.read_text(encoding="utf-8"), path.parent, depth + 1)

    return _INPUT.sub(sub, text)


@cache
def paper_tex(slug: str) -> str:
    """One paper's source with its shared inputs expanded."""
    path = paper_path(slug)
    return _expand(path.read_text(encoding="utf-8"), path.parent)


@cache
def all_tex() -> str:
    """Every paper's source, concatenated in series order, shared inputs expanded.

    A `%` comment line is inserted between papers so no pinned pattern can match across a
    boundary and silently stitch two papers' sentences into one claim.
    """
    parts = []
    for slug in PAPERS:
        parts.append(f"% ===== BEGIN {slug} =====")
        parts.append(paper_tex(slug))
        parts.append(f"% ===== END {slug} =====")
    return "\n".join(parts)


def papers_containing(needle: str) -> list[str]:
    """Which papers contain `needle` as literal text. Used by the cross-paper agreement checks."""
    return [slug for slug in PAPERS if needle in paper_tex(slug)]


def main() -> int:  # pragma: no cover - CLI convenience
    for slug in PAPERS:
        tex = paper_tex(slug)
        print(
            f"{PAPER_NUMERALS[slug]:>3}  {slug:26s} "
            f"{len(tex.split()):6d} words  {PAPER_TITLES[slug]}"
        )
    print(f"     {'concatenated':26s} {len(all_tex().split()):6d} words")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
