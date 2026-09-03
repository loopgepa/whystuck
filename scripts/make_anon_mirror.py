"""Build the anonymous artifact mirror a double-blind submission has to point at.

The three papers claim an artifact release, and every venue that fits them is double-blind, so the
link in the abstract cannot be this repository: every commit here is authored by the same name and
address, and the project name resolves to it in one search. `SUBMISSION.md` records why the
mirror is a larger job than the paper build, and why it has to be rebuilt at the commit being
submitted. This is that job, minus the one step that
is not a program's to take: it writes a verified tree and stops. Publishing it is the author's.

What makes it more than a search-and-replace, and how each part is handled:

* HISTORY LEAKS ON EVERY COMMIT, so the mirror is a snapshot. The file list comes from
  `git ls-files`, and no `.git` is copied.
* THE PROJECT NAME IS ITSELF IDENTIFYING. It is in the arXiv packager's own token list, because a
  reviewer who searches it finds the repository. So the package is renamed throughout, not scrubbed
  out. 856 occurrences across 164 tracked files, none of them inside a run JSON, a replicate
  scoring or a cassette, which is why the rename cannot break replay. That was checked before this
  script was written rather than hoped for.
* THE REPRODUCIBILITY APPARATUS IS THE POINT, so it stays: the run JSONs, the per-replicate
  scorings and the cassettes are copied verbatim.
* SOME FILES NAME THE AUTHOR IRREDUCIBLY and a mirror does not need them: a code of conduct with a
  contact address, the submission notes, the arXiv packager and its own token list. They are
  dropped by name rather than scrubbed, because a scrubbed code of conduct is worse than none.
* THE PAPERS THEMSELVES ARE NOT ARTIFACTS. A reviewer gets the PDF from the venue. Shipping
  `docs/papers/` would put the author block, the corrections registers and the pre-split monolith
  into the mirror for no gain.

The last step is a byte-level scan for every identifying token, over every file written, and a
refusal if one survives. Bytes rather than parsed text, because the leak that motivated the same
check in the paper packager was a LaTeX comment: it renders as nothing and ships anyway.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_arxiv_package import IDENTIFYING  # noqa: E402

#: Paths a mirror does not need and cannot be made anonymous by substitution. Dropped whole.
DROP = (
    # Not all of docs/papers/. The paper SOURCES stay, because `verify_paper_claims.py` scrapes
    # them to re-derive every claim against the artifacts, and that script is the most useful
    # thing in the release for a reviewer. Dropping the sources to protect anonymity would remove
    # the checker with them. They go through the substitution pass instead, and the byte scan is
    # what decides whether that was enough.
    "docs/papers/SUBMISSION.md",  # names the author, and discusses this very step
    "docs/papers/ENDORSEMENT-REQUEST-DRAFT.md",
    "docs/papers/PRE-SPLIT-SOURCE.tex",  # the monolith, superseded and not needed
    "docs/papers/README.md",  # describes the carve, names the author
    "CODE_OF_CONDUCT.md",  # contact address
    "SECURITY.md",  # contact address
    "CONTRIBUTING.md",  # names the org's review process
    "LAUNCH_CHECKLIST.md",
    "DECISIONS.md",
    "docs/design-records.md",
    ".claude-plugin/",
    ".github/",
    "scripts/make_arxiv_package.py",
    "scripts/make_anon_mirror.py",
    "scripts/setup_branch_protection.sh",
    "tests/test_arxiv_package.py",
    "tests/test_anonymous_package.py",
    "server.json",
    "mkdocs.yml",
    "CLAUDE.md",
    "AGENTS.md",
)


def is_text(blob: bytes) -> bool:
    """Whether to rewrite this file, decided by decodability rather than by extension.

    An extension allow-list let four files through the scan: `.gitignore` and the pre-commit hook
    have no suffix at all, and two terminal-recording `.tape` files were not on the list. Every one
    of them still named the project. Decodability catches the whole class instead of growing the
    list one leak at a time.
    """
    if b"\x00" in blob[:8192]:
        return False
    try:
        blob.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


#: Substitutions applied to text files, longest first so a shorter token cannot pre-empt a longer.
def substitutions(name: str) -> list[tuple[str, str]]:
    return [
        ("sysknife-development@protonmail.com", "anonymous@example.com"),
        ("github.com/loopgepa/whystuck", f"example.com/anonymous/{name}"),
        ("loopgepa/whystuck", f"anonymous/{name}"),
        ("Vladimir Rotariu", "Anonymous"),
        ("vladimirrott", "anonymous"),
        ("loopgepa", "anonymous"),
        ("Rotariu", "Anonymous"),
        ("Vladimir", "Anonymous"),
        ("Independent researcher", "Anonymous"),
        ("WhyStuck", name.capitalize()),
        ("whystuck", name),
        ("WHYSTUCK", name.upper()),
    ]


def tracked(repo: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def keep(rel: str) -> bool:
    return not any(rel == d or rel.startswith(d) for d in DROP)


def retarget(rel: str, name: str) -> str:
    """The path a file takes in the mirror, with the package directory renamed."""
    return (
        rel.replace("src/whystuck", f"src/{name}").replace("whystuck/", f"{name}/", 1)
        if rel.startswith("src/whystuck")
        else rel
    )


def leaks(tree: Path) -> list[tuple[str, str]]:
    """(file, token) for every identifying string surviving anywhere in the written tree."""
    found: list[tuple[str, str]] = []
    needles = [(t, t.lower().encode()) for t in IDENTIFYING]
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        blob = path.read_bytes().lower()
        for token, needle in needles:
            if needle in blob:
                found.append((str(path.relative_to(tree)), token))
    return found


def build(repo: Path, dest: Path, name: str) -> tuple[int, int]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subs = substitutions(name)
    written = rewritten = 0
    for rel in tracked(repo):
        if not keep(rel):
            continue
        src = repo / rel
        if not src.is_file():
            continue
        out = dest / retarget(rel, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = src.read_bytes()
        if is_text(blob):
            body = blob.decode("utf-8")
            before = body
            for old, new in subs:
                body = body.replace(old, new)
            out.write_text(body, encoding="utf-8")
            rewritten += before != body
        else:
            shutil.copy2(src, out)
        written += 1
    return written, rewritten


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--name", default="loopstudy", help="the neutral package name")
    args = ap.parse_args()

    if re.search(r"|".join(re.escape(t) for t in IDENTIFYING), args.name, re.I):
        raise SystemExit(f"--name {args.name!r} contains an identifying token")

    written, rewritten = build(args.repo, args.dest, args.name)
    surviving = leaks(args.dest)

    print(f"{written} file(s) written to {args.dest}, {rewritten} rewritten")
    for area, pattern in (
        # No hyphen after `study`. With one, this glob excluded `docs/validation-study.json` --
        # the flagship run, whose fifteen arm-runs carry the 15/15 statistic all three papers
        # lead with -- so the line attesting that the reproducibility apparatus survived the
        # rename said 42 while the mirror held 43. The file was always copied; only the
        # attestation omitted it, which is the worse direction for a number whose job is to say
        # nothing was lost.
        ("run JSONs", "docs/validation-study*.json"),
        ("replicate scorings", "docs/repeat/*.json"),
        ("cassettes", "tests/fixtures/cassettes/**/*.json"),
    ):
        print(f"  {len(list(args.dest.glob(pattern))):4} {area}")  # glob, so nesting counts

    if surviving:
        print(f"\nREFUSED: {len(surviving)} identifying string(s) survive:", file=sys.stderr)
        for path, token in surviving[:20]:
            print(f"  {path}: {token}", file=sys.stderr)
        raise SystemExit(1)
    print("\nOK  no identifying token survives anywhere in the written tree")
    print("    The tree is a snapshot with no history. Publishing it is the author's step.")


if __name__ == "__main__":
    main()
