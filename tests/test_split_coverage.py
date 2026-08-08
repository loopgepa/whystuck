"""Guard the three-way paper split.

One 46-page draft became three papers. The split's risk is not that it breaks a build -- LaTeX
would say so -- but that a claim quietly ceases to exist: a paragraph belonging to no paper's
argument gets dropped, and nothing anywhere errors. `scripts/verify_split_coverage.py` compares the
frozen pre-split source against the three papers and requires every quantity to survive; these tests
keep that comparison honest, in the two ways it could go wrong.

It could pass by having nothing to compare (a missing or truncated baseline, an empty paper list),
so `test_the_check_is_not_vacuous` and `test_the_baseline_is_the_whole_draft` deny that. And it
could pass by exempting whatever fails, so `test_the_deliberate_drop_list_does_not_rot` requires
every exemption to be both necessary and still needed. The drop list is currently empty, which is
the strongest form of the result and also the easiest state to erode.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import paper_sources  # noqa: E402
import verify_split_coverage as vsc  # noqa: E402


def test_every_pre_split_quantity_survives():
    """The proof itself: no quantity in the pre-split draft is absent from all three papers."""
    lost = vsc.missing_quantities()
    assert not lost, (
        f"{len(lost)} quantities present before the split are in no paper after it: "
        + ", ".join(sorted(lost))
    )


def test_the_deliberate_drop_list_does_not_rot():
    """Every exemption must be necessary (the number really is gone) and real (it really was there).

    Without this the list becomes a place to silence failures: an entry added for one edit keeps
    suppressing the check after a later edit restores the number, and the coverage claim
    quietly weakens with no diff to notice.
    """
    assert not vsc.stale_drops(), "\n".join(vsc.stale_drops())


def test_every_superseded_value_was_replaced_not_deleted():
    """The corrected-statistic route must prove replacement in both directions.

    A value a corrected statistic superseded is allowed to vanish only if its replacement is
    present. Otherwise `SUPERSEDED` would be an exemption list wearing a better name, and a
    genuinely lost quantity could be hidden by adding one line to it.
    """
    assert not vsc.broken_supersessions(), "\n".join(vsc.broken_supersessions())


def test_the_superseded_map_is_not_a_blanket_exemption():
    """Every entry must name a concrete replacement, and no entry may map a value to itself."""
    for old, replacements in vsc.SUPERSEDED.items():
        news = replacements.split()
        assert news, f"{old}: no replacement named"
        assert old not in news, f"{old}: maps to itself"
        for new in news:
            assert vsc._QUANTITY.fullmatch(new), f"{old} -> {new!r} is not a quantity"


def test_the_estimator_fix_actually_landed_in_the_code():
    """The 47 superseded values exist because of one fix; if the fix were reverted they would come
    back and this file would still pass, so the fix itself is asserted here.

    A population sd would understate sigma by sqrt(K/(K-1)) and inflate every t against t(K-1).
    """
    import statistics

    import repeat_scores as rc

    draws = [0.0, 0.1, 0.2, 0.3, 0.4]
    result = rc.LevelResult(level="probe", k=5, replicate_deltas=draws, n_tasks=30)
    assert result.sd_between_replicates == pytest.approx(statistics.stdev(draws))
    assert result.sd_between_replicates != pytest.approx(statistics.pstdev(draws))
    assert result.df == len(draws) - 1


def test_the_drop_list_entries_carry_a_reason():
    """A bare exemption is indistinguishable from a lost claim six months later."""
    for quantity, reason in vsc.DELIBERATE_DROPS.items():
        assert len(reason) > 40, f"{quantity}: reason too short to be a reason ({reason!r})"


def test_every_spelled_out_count_survives():
    """Counts the papers state in words rather than digits, which the numeric scan cannot see."""
    assert not vsc.missing_spelled()


def test_every_spelled_supersession_holds_in_both_directions():
    """A spelled count may change, but only on the record and only if the replacement landed.

    Without this, `SPELLED_SUPERSEDED` would be a plain exemption list: an entry could sit there
    naming a replacement that never made it into any paper, and the count would be gone with a
    note claiming otherwise.
    """
    assert not vsc.broken_spelled_supersessions()


def test_the_check_is_not_vacuous():
    """Dropping a paper's worth of text must be reported as loss.

    The comparison is a set difference, so it passes trivially if the "after" side is accidentally
    the same text as the "before" side, or if the baseline is empty. Removing the largest paper
    from the series must therefore produce failures, and this asserts it does.
    """
    before = vsc.quantities(vsc.frozen_tex())
    without_p1 = vsc.quantities(
        "\n".join(
            paper_sources.paper_tex(s) for s in paper_sources.PAPERS if s != "p1-conditional-lever"
        )
    )
    assert before - without_p1, (
        "removing paper I lost no quantity, which means the comparison is not actually reading "
        "the papers"
    )


def test_the_baseline_is_the_whole_draft():
    """A truncated baseline would make the proof pass by asking less.

    Pinned to the pre-split draft's structure rather than a byte count, so ordinary whitespace
    edits do not break it but a lost half of the file does.
    """
    frozen = vsc.frozen_tex()
    assert frozen.count("\\section{") >= 14, "frozen baseline has lost sections"
    assert "\\end{document}" in frozen, "frozen baseline is truncated"
    assert len(vsc.quantities(frozen)) > 300, "frozen baseline has lost most of its quantities"


def test_the_frozen_baseline_matches_the_recorded_commit():
    """The baseline is byte-identical to `docs/paper/main.tex` at the commit it names.

    This is the one assertion that stops the proof being circular: without it, a quantity could be
    deleted from the papers and from the baseline in the same commit, and everything would pass.
    Skipped only where git history is genuinely unreachable (a source tarball), and skipped loudly.
    """
    try:
        recorded = subprocess.run(
            ["git", "show", f"{vsc.FROZEN_COMMIT}:docs/paper/main.tex"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"git history unreachable, cannot verify the frozen baseline: {exc}")
    assert vsc.frozen_tex() == recorded, (
        f"PRE-SPLIT-SOURCE.tex differs from {vsc.FROZEN_COMMIT[:8]}:docs/paper/main.tex. "
        "The baseline must not be edited -- edit the papers instead."
    )


def test_the_frozen_baseline_is_never_built():
    """It is provenance, not a paper. If it acquired a build it would drift into a fourth paper."""
    assert vsc.FROZEN.name == "PRE-SPLIT-SOURCE.tex"
    assert vsc.FROZEN.parent == pathlib.Path("docs/papers")
    assert vsc.FROZEN.name not in [paper_sources.paper_path(s).name for s in paper_sources.PAPERS]
    for sibling in ("PRE-SPLIT-SOURCE.pdf", "PRE-SPLIT-SOURCE.aux", "PRE-SPLIT-SOURCE.log"):
        assert not (vsc.FROZEN.parent / sibling).exists(), f"{sibling} should not be built"


def test_comment_stripping_does_not_count_source_comments_as_claims():
    """A number that appears only in a `%` comment is not a claim, and an escaped `\\%` is not a
    comment. Both directions matter: the first would inflate the baseline with plumbing, the second
    would truncate real sentences at the first percent sign."""
    assert vsc.quantities("% only 1.234 in a comment") == set()
    assert vsc.quantities("the rate is 57.5\\% of tasks") == {"57.5"}
    assert vsc.quantities("value 0.128 % but not 9.999") == {"0.128"}


def test_all_three_papers_carry_a_substantial_share():
    """A split where one paper carried almost nothing would be a relabelling, not a split."""
    per_paper = vsc.coverage_by_paper()
    assert len(per_paper) == 3
    for slug, count in per_paper.items():
        assert count >= 50, f"{slug} carries only {count} of the pre-split quantities"


def test_shared_inputs_are_expanded_not_skipped():
    """A number in `shared/loop.tex` is a number every paper's reader sees.

    If `\\input` were left unexpanded, every guard reading these sources would under-count coverage
    and the shared harness section's claims would silently stop being checked.
    """
    shared = pathlib.Path("docs/papers/shared/loop.tex").read_text(encoding="utf-8")
    marker = "registers twenty-six committed runs"
    assert marker in shared
    for slug in paper_sources.PAPERS:
        tex = paper_sources.paper_tex(slug)
        assert marker in tex, f"{slug} did not expand shared/loop.tex"
        assert "\\input{" not in tex, f"{slug} still contains an unexpanded \\input"


def test_a_missing_shared_input_raises_rather_than_resolving_to_nothing(tmp_path):
    """An unresolvable `\\input` must fail loudly. Silently yielding the un-expanded directive is
    how a whole shared section would stop being covered while every check still passed."""
    stray = tmp_path / "main.tex"
    stray.write_text("\\input{../shared/does-not-exist}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        vsc.paper_sources._expand(stray.read_text(), stray.parent)


def test_paper_boundaries_cannot_be_matched_across():
    """The concatenation separates papers with a comment line, so no pinned pattern can stitch the
    tail of one paper to the head of the next and validate a claim neither paper makes."""
    joined = paper_sources.all_tex()
    for slug in paper_sources.PAPERS:
        assert f"% ===== BEGIN {slug} =====" in joined
        assert f"% ===== END {slug} =====" in joined
    # Every paper's document environment closes before the next one opens.
    assert joined.count("\\end{document}") == len(paper_sources.PAPERS)
    order = [m.group(1) for m in re.finditer(r"% ===== BEGIN (\S+) =====", joined)]
    assert tuple(order) == paper_sources.PAPERS, "concatenation order is not the series order"
