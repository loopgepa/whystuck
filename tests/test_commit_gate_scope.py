r"""The pre-commit gate must fire on the files whose published numbers it guards.

WHY THIS EXISTS. `.githooks/pre-commit` runs `verify_paper_claims.py` only when the staged paths
look like they could move a published number. That scoping is deliberate: the check replays
cassettes, and a hook that is slow on unrelated commits gets disabled, which is worse than a slow
one. But the scope is a regex, and a scope regex that stops matching is precisely the silent no-op
this series is about -- the hook prints `all green` having skipped the one check that mattered.

It had already happened. The pattern named `docs/paper/`, the single-file path that existed before
the 46-page draft was split into `docs/papers/p1-conditional-lever/main.tex` and its siblings.
`scripts/paper_sources.py` exists because every claim checker and coverage guard had to be migrated
off that hard-coded path; its own docstring names the risk of leaving one behind. The hook's scope
regex was the one left behind, so from the split until this test, a commit touching only paper prose
skipped the claim check entirely. That is the exact commit shape that printed a `PASS` rate of $1\%$
against an artifact measuring $0.5\%$ -- a `.tex`-only edit.

The paths come from `paper_sources`, never from a list written out here: a guard carrying its own
copy of the three paper paths would drift away from the papers the same way the hook did.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_sources  # noqa: E402

_HOOK = ROOT / ".githooks" / "pre-commit"

#: Frozen by design and never built, with its own guard forbidding edits, so whether the claim
#: gate fires on it is not a property worth pinning either way.
_FROZEN = "docs/papers/PRE-SPLIT-SOURCE.tex"


def _scope_pattern() -> str:
    """Pull the scope regex out of the hook instead of restating it.

    Restating it here would let the hook narrow while this file kept asserting against the old
    pattern -- a guard that passes by testing a string nobody runs.
    """
    text = _HOOK.read_text(encoding="utf-8")
    # Anchored on the gate that actually guards the claim check, not on there being only one
    # gate in the file. It was "exactly one" until a second, independent gate was added for a
    # different checker, at which point uniqueness stopped identifying anything -- and loosening
    # this to "the first one found" would have left the guard pinning whichever gate happened to
    # come first in the file.
    blocks = [b for b in text.split("\nif ") if "verify_paper_claims" in b]
    assert len(blocks) == 1, (
        f"expected exactly one `if` block running verify_paper_claims in {_HOOK.name}, found "
        f"{len(blocks)}. This guard can no longer tell which pattern gates the claim check."
    )
    found = re.findall(r"grep -qE '([^']+)'", blocks[0])
    assert len(found) == 1, (
        f"expected exactly one `grep -qE '...'` scope pattern in the block gating "
        f"verify_paper_claims, found {len(found)}: {found}."
    )
    return found[0]


def _fires_on(path: str) -> bool:
    """Ask the engine the hook actually uses.

    `grep -E` is POSIX ERE and Python's `re` is not. They agree on this pattern today, so a
    Python-side match would give the right answer for the wrong reason and stop being evidence
    the moment the pattern reached for a dialect-specific construct.
    """
    done = subprocess.run(
        ["grep", "-qE", _scope_pattern()],
        input=f"{path}\n",
        text=True,
        capture_output=True,
    )
    assert done.returncode in (0, 1), f"grep failed on {path!r}: {done.stderr}"
    return done.returncode == 0


def _paper_sources() -> list[str]:
    """Every `.tex` a published number can be printed from, the frozen pre-split copy aside."""
    found = sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / paper_sources.PAPERS_DIR).rglob("*.tex")
    )
    return [p for p in found if p != _FROZEN]


@pytest.mark.parametrize("slug", paper_sources.PAPERS)
def test_the_gate_fires_on_each_paper_main_file(slug: str) -> None:
    path = str(paper_sources.paper_path(slug))
    assert (ROOT / path).is_file(), (
        f"{path} does not exist, so this guard is asserting about a file that moved; "
        f"update paper_sources.PAPERS"
    )
    assert _fires_on(path), (
        f"a commit editing only {path} would skip verify_paper_claims. The gate pattern is "
        f"{_scope_pattern()!r}, which does not match it."
    )


def test_the_gate_fires_on_the_shared_inputs_too() -> None:
    """A number in `shared/loop.tex` is a number every paper prints."""
    shared = [p for p in _paper_sources() if "/shared/" in p]
    assert shared, "no shared .tex inputs found; this assertion has lost its subject"
    missed = [p for p in shared if not _fires_on(p)]
    assert not missed, f"editing these alone would skip the claim check: {missed}"


def test_the_gate_fires_on_every_paper_tex() -> None:
    """The sweep, so a fourth paper or a new section file is covered without editing this file."""
    missed = [p for p in _paper_sources() if not _fires_on(p)]
    assert not missed, f"paper sources outside the claim gate: {missed}"


@pytest.mark.parametrize(
    "path",
    [
        "docs/validation-study-c2.json",
        "docs/repeat/curve-run10-s0-k0.json",
        "scripts/verify_paper_claims.py",
        "tests/fixtures/cassettes/anything.json",
        "src/whystuck/eval/study.py",
    ],
)
def test_the_gate_still_fires_on_artifacts_and_analysis(path: str) -> None:
    """Regression: widening the pattern to the papers must not drop what it already caught."""
    assert _fires_on(path), f"{path} no longer triggers the claim check"


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/whystuck/core/storage.py",
        "tests/test_diff.py",
        "docs/getting-started.md",
        ".githooks/pre-commit",
    ],
)
def test_the_gate_does_not_fire_on_everything(path: str) -> None:
    """Non-vacuity. A pattern of `.*` would satisfy every other test in this file.

    It would also make the check unconditional, undoing the scoping the hook documents as the
    reason the gate is survivable at all.
    """
    assert not _fires_on(path), (
        f"{path} triggers the cassette-replaying claim check; the gate is no longer scoped"
    )


def test_the_hook_names_a_directory_that_exists() -> None:
    """The defect this file was written for, stated directly.

    The pattern named `docs/paper/` after the tree had only `docs/papers/`. Every literal
    directory prefix in the gate must resolve, or the alternative it sits in is dead.
    """
    pattern = _scope_pattern()
    prefixes = re.findall(r"\^\(?([A-Za-z0-9_./-]*/)", pattern) + re.findall(
        r"\|([A-Za-z0-9_./-]*/)", pattern
    )
    assert prefixes, f"no literal directory prefixes parsed out of {pattern!r}"
    dead = [p for p in prefixes if not (ROOT / p).is_dir()]
    assert not dead, f"the gate keys on directories that do not exist: {dead}"


# --------------------------------------------------------------------- the papyrustrail gate
#
# The same silent-no-op risk as above, for the second gate: a scope regex that stops matching
# leaves the hook printing `all green` having skipped a check. That gate is optional (the tool
# lives in a separate project and not every machine has it), which makes a narrowing MORE likely
# to go unnoticed, not less.


def _papyrustrail_scope() -> str:
    text = _HOOK.read_text(encoding="utf-8")
    blocks = [b for b in text.split("\nif ") if "papyrustrail verify" in b]
    assert len(blocks) == 1, (
        f"expected exactly one `if` block running papyrustrail, found {len(blocks)}"
    )
    found = re.findall(r"grep -qE '([^']+)'", blocks[0])
    assert len(found) == 1, f"expected one scope pattern in the papyrustrail block, found {found}"
    return found[0]


def _papyrustrail_fires_on(path: str) -> bool:
    done = subprocess.run(
        ["grep", "-qE", _papyrustrail_scope()], input=f"{path}\n", text=True, capture_output=True
    )
    assert done.returncode in (0, 1), f"grep failed on {path!r}: {done.stderr}"
    return done.returncode == 0


@pytest.mark.parametrize("slug", paper_sources.PAPERS)
def test_the_papyrustrail_gate_fires_on_each_paper(slug: str) -> None:
    assert _papyrustrail_fires_on(str(paper_sources.paper_path(slug)))


@pytest.mark.parametrize(
    "path",
    [
        "docs/validation-study-c2.json",
        "docs/repeat/L1-B-s0-k0.json",
        "papyrustrail.toml",
        ".papyrustrail/baseline.json",
    ],
)
def test_the_papyrustrail_gate_fires_on_what_it_reads(path: str) -> None:
    """Its config and its baseline included: a commit editing only the baseline must re-verify,
    or an accepted-away finding could land without the check that would have caught it."""
    assert _papyrustrail_fires_on(path)


@pytest.mark.parametrize(
    "path",
    ["README.md", "docs/getting-started.md", "src/whystuck/core/storage.py", "tests/test_diff.py"],
)
def test_the_papyrustrail_gate_does_not_fire_on_everything(path: str) -> None:
    """A gate that fires on every commit is a gate that gets disabled."""
    assert not _papyrustrail_fires_on(path)
