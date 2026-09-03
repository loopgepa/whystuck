"""The anonymous mirror builder, and the one property that makes its verdict mean anything.

A scanner that finds nothing anywhere passes every tree, so the first test below plants the
author in a staged file and requires the scan to find him. The rest pin the decisions that were
made after watching them fail: text detection by decodability rather than by extension, because an
extension allow-list let `.gitignore`, a git hook and two `.tape` files through still naming the
project; and the paper sources staying in the mirror, because the claim checker scrapes them and
dropping them would remove the most useful script in the release.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_anon_mirror as mirror  # noqa: E402
from make_anon_mirror import (  # noqa: E402
    DROP,
    build,
    is_text,
    keep,
    leaks,
    retarget,
    substitutions,
)


def test_the_scanner_finds_the_author_when_he_is_there(tmp_path: Path) -> None:
    """Anti-vacuity. Without this, a scan that always returns empty reads as a clean tree."""
    (tmp_path / "planted.md").write_text("by Vladimir Rotariu, github.com/loopgepa/whystuck\n")
    found = leaks(tmp_path)
    assert found, "the scan must find a planted author"
    assert {token for _, token in found} >= {"Vladimir", "Rotariu", "loopgepa", "whystuck"}


def test_the_scanner_is_case_insensitive_and_reads_bytes(tmp_path: Path) -> None:
    """The leak that motivated the same check in the paper packager was a LaTeX comment."""
    (tmp_path / "c.tex").write_text("% WHYSTUCK, in a comment that renders as nothing\n")
    assert [t for _, t in leaks(tmp_path)] == ["whystuck"]


def test_text_is_decided_by_decodability_not_extension() -> None:
    assert is_text(b"#!/usr/bin/env bash\necho whystuck\n"), "an extensionless hook is text"
    assert not is_text(b"\x89PNG\r\n\x1a\n\x00\x00"), "a PNG is not"
    assert not is_text(b"\xff\xfe\x00binary"), "undecodable bytes are not"


def test_the_paper_sources_stay_and_the_submission_notes_go() -> None:
    """The mirror keeps what a reviewer re-derives claims with, and drops what names the author."""
    assert keep("docs/papers/p1-conditional-lever/main.tex")
    assert keep("docs/papers/shared/loop.tex")
    assert keep("scripts/verify_paper_claims.py")
    assert not keep("docs/papers/SUBMISSION.md")
    assert not keep("docs/papers/PRE-SPLIT-SOURCE.tex")
    assert not keep("CODE_OF_CONDUCT.md")


def test_the_package_directory_is_renamed_not_scrubbed() -> None:
    """The project name is itself identifying, so it is replaced rather than removed."""
    assert retarget("src/whystuck/eval/study.py", "loopstudy") == "src/loopstudy/eval/study.py"
    assert retarget("docs/index.md", "loopstudy") == "docs/index.md"


def test_longer_tokens_substitute_before_the_shorter_ones_inside_them() -> None:
    """ "Vladimir Rotariu" must not become "Anonymous Anonymous" via two passes."""
    body = "Vladimir Rotariu <sysknife-development@protonmail.com>"
    for old, new in substitutions("loopstudy"):
        body = body.replace(old, new)
    assert body == "Anonymous <anonymous@example.com>"


def test_a_dropped_path_is_matched_as_a_prefix_not_a_substring() -> None:
    assert not keep("docs/papers/SUBMISSION.md")
    assert keep("docs/papers/p1-conditional-lever/SUBMISSION-notes-that-do-not-exist.md")
    assert all(not d.startswith("/") for d in DROP), "DROP holds repo-relative paths"


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git index for the file list")
def test_the_attestation_counts_every_run_json_including_the_bare_named_one() -> None:
    """The printed summary is the attestation that nothing was lost, so its glob has to be right.

    Source-level because the count lives in a `print` inside `main()`. `docs/validation-study.json`
    is the flagship run whose fifteen arm-runs carry the 15/15 statistic all three papers lead
    with, and `validation-study-*.json` is exactly the pattern that omits it.
    """
    src = inspect.getsource(mirror)
    assert '"docs/validation-study*.json"' in src, (
        "the run-JSON attestation glob must not have a hyphen after `study`, or it under-reports "
        "the release by dropping docs/validation-study.json"
    )
    assert '"docs/validation-study-*.json"' not in src


def test_a_built_mirror_carries_the_artifacts_and_leaks_nothing(tmp_path: Path) -> None:
    """The whole point: the reproducibility apparatus survives and the identity does not."""
    written, _ = build(ROOT, tmp_path / "m", "loopstudy")
    tree = tmp_path / "m"
    assert written > 500, "the mirror must carry the release, not a subset of it"
    assert len(list(tree.glob("docs/repeat/*.json"))) == 225
    assert len(list(tree.glob("tests/fixtures/cassettes/**/*.json"))) == 267
    # Run JSONs were the one artifact class with no assertion here, which is how the attestation
    # line in `make_anon_mirror` kept a `validation-study-*.json` glob: the hyphen drops the
    # bare-named flagship run, so it printed 42 for a tree holding 43. Compared by NAME against
    # the source rather than against a literal, since a literal would have pinned the bug.
    source_runs = sorted(q.name for q in ROOT.glob("docs/validation-study*.json"))
    assert len(source_runs) > 40, "the source glob found almost no runs; this would be vacuous"
    assert "validation-study.json" in source_runs, (
        "the bare-named flagship run is the file the hyphenated glob dropped; without it in the "
        "source this assertion no longer covers the case it was written for"
    )
    assert sorted(q.name for q in tree.glob("docs/validation-study*.json")) == source_runs
    assert (tree / "src" / "loopstudy" / "eval" / "cassette.py").is_file()
    assert not (tree / "src" / "whystuck").exists()
    assert leaks(tree) == []
