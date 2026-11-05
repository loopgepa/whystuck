r"""The three findings from the papers, as things the tool actually reports.

Each is a result the series paid for and the served tool did not expose. They were knowledge in
`src/whystuck/eval/` (the research harness) and absent from the surface a user meets.

1. **p0, the base rate of already-correct first drafts.** At p0 = 0.067 Paper I's search
   converged on a verifier that never accepts and loop-level optimization bought nothing
   measurable. Its one positive cell at 0.267 is not distinguishable from the spread of nine cells
   at 0.067 and did not reproduce on a third worker, so Paper I withdrew the claim that the answer
   depends on p0, and the tool must not recommend loop-level optimization on that ground.
2. **The degenerate-verifier screen.** That failure is visible as a PASS rate of zero on the
   optimizer's output, which costs nothing to check and is the cheapest result in the series.
3. **The silent no-op.** Paper III: in the flagship superseded run the critique gene was
   byte-identical to the seed in 15/15 winning genomes while `generate` moved in 5/15. An
   optimizer that left the treatment gene alone and reports a score is the failure mode the paper
   is named for, which is why the screen answers per component.
"""

from __future__ import annotations

import pytest

from whystuck.core.diagnosis import (
    LOOP_LEVER_HIGH,
    LOOP_LEVER_LOW,
    LOOP_LEVER_MID,
    loop_lever_verdict,
    p0_base_rate,
    screen_optimizer_result,
)
from whystuck.core.models import LoopMetrics, LoopTrace


def _trace(outcome: str, retries: int) -> LoopTrace:
    return LoopTrace(
        id=f"t{retries}{outcome}",
        loop_id="L",
        created_at="2026-09-10T00:00:00Z",
        final_outcome=outcome,  # type: ignore[arg-type]
        metrics=LoopMetrics(retry_count=retries),
    )


def test_p0_counts_only_first_drafts_that_needed_no_repair() -> None:
    traces = [
        _trace("success", 0),
        _trace("success", 2),
        _trace("failure", 1),
        _trace("failure", 0),
    ]
    assert p0_base_rate(traces) == 0.25


def test_p0_is_none_rather_than_zero_when_there_are_no_traces() -> None:
    """A base rate of 0.0 and 'nobody has measured it' are different recommendations."""
    assert p0_base_rate([]) is None


def test_a_low_base_rate_advises_against_loop_level_optimization() -> None:
    verdict = loop_lever_verdict(0.067)
    assert verdict["regime"] == LOOP_LEVER_LOW
    assert verdict["loop_optimization_indicated"] is False
    assert "0.067" in verdict["evidence"]


def test_the_low_regime_evidence_carries_no_withdrawn_claim() -> None:
    """Paper I withdrew the low-p0 cost and its Goodhart reading: the nine cells at p0 = 0.067
    average +0.006, so the evidence may say 'no measurable gain', never 'negative' or 'Goodhart'."""
    evidence = str(loop_lever_verdict(0.067)["evidence"]).lower()
    assert "goodhart" not in evidence
    assert "negative" not in evidence
    assert "no measurable" in evidence


@pytest.mark.parametrize("p0", [0.267, 0.30, 0.667, 0.9])
def test_a_higher_base_rate_does_not_claim_the_withdrawn_gain(p0: float) -> None:
    """Above the low regime the release holds one positive cell, not an established gain: the
    tool reports the regime and declines to recommend either way."""
    verdict = loop_lever_verdict(p0)
    assert verdict["regime"] in (LOOP_LEVER_MID, LOOP_LEVER_HIGH)
    assert verdict["loop_optimization_indicated"] is None
    evidence = str(verdict["evidence"]).lower()
    assert "swing" not in evidence and "stopped losing" not in evidence
    assert "not established" in evidence


def test_the_regime_boundaries_still_sit_between_the_measured_levels() -> None:
    assert loop_lever_verdict(0.149)["regime"] == LOOP_LEVER_LOW
    assert loop_lever_verdict(0.15)["regime"] == LOOP_LEVER_MID
    assert loop_lever_verdict(0.50)["regime"] == LOOP_LEVER_HIGH


def test_an_unmeasured_base_rate_refuses_to_advise_either_way() -> None:
    verdict = loop_lever_verdict(None)
    assert verdict["loop_optimization_indicated"] is None
    assert "not measured" in verdict["evidence"].lower()


def test_every_regime_points_at_the_fix_side() -> None:
    """Paper II holds at every p0: of a perfect oracle's +0.0437, the verdict is +0.0110 and the
    repair text +0.0327, so the advice does not flip with the regime."""
    for p0 in (None, 0.0, 0.067, 0.30, 0.9):
        assert "repair" in loop_lever_verdict(p0)["advice"].lower()


def test_an_optimizer_that_changed_nothing_is_reported_as_a_no_op() -> None:
    warnings = screen_optimizer_result(genes_before={"verify": "A"}, genes_after={"verify": "A"})
    assert any("no-op" in w for w in warnings), warnings


def test_an_optimizer_that_changed_something_is_not_flagged_as_a_no_op() -> None:
    warnings = screen_optimizer_result(genes_before={"verify": "A"}, genes_after={"verify": "B"})
    assert not any("no-op" in w for w in warnings), warnings


def test_a_verifier_that_never_accepts_is_screened() -> None:
    warnings = screen_optimizer_result(
        genes_before={"verify": "A"}, genes_after={"verify": "B"}, verify_pass=0, verify_total=30
    )
    assert any("never accepts" in w for w in warnings), warnings


def test_a_verifier_that_accepts_sometimes_is_not_screened() -> None:
    warnings = screen_optimizer_result(
        genes_before={"verify": "A"}, genes_after={"verify": "B"}, verify_pass=4, verify_total=30
    )
    assert not any("never accepts" in w for w in warnings), warnings


def test_the_screen_says_nothing_when_it_was_given_nothing_to_look_at() -> None:
    """Separating 'asked, no objection' from 'could not ask': an unmeasured PASS rate must not
    read as a clean one."""
    warnings = screen_optimizer_result(genes_before={"verify": "A"}, genes_after={"verify": "B"})
    assert not any("never accepts" in w for w in warnings)
    assert any("pass rate not measured" in w.lower() for w in warnings), warnings


@pytest.mark.parametrize("total", [0, None])
def test_a_zero_or_missing_denominator_does_not_divide(total: int | None) -> None:
    warnings = screen_optimizer_result(
        genes_before={"v": "A"}, genes_after={"v": "B"}, verify_pass=0, verify_total=total
    )
    assert any("pass rate not measured" in w.lower() for w in warnings), warnings


def test_a_treatment_gene_that_did_not_move_is_reported_per_component() -> None:
    """The flagship run's shape: `generate` changed and the critique did not. A whole-genome
    "did anything change" says yes there, which is the reading Paper III says does not work, so
    the screen has to answer per component."""
    before = {"generate": "a", "ensure_correct": "b"}
    after = {"generate": "x", "ensure_correct": "b"}
    named = screen_optimizer_result(before, after, treatment_genes=["ensure_correct"])
    assert any("ensure_correct" in w and "byte-identical" in w for w in named), named
    unnamed = screen_optimizer_result(before, after)
    assert any("ensure_correct" in w and "byte-identical" in w for w in unnamed), unnamed
    moved = screen_optimizer_result(before, {"generate": "x", "ensure_correct": "y"})
    assert not any("byte-identical" in w for w in moved), moved


def test_a_blanked_gene_is_reported_even_though_it_counts_as_changed() -> None:
    """Defect 8: an empty completion accepted as a gene differs from the seed, so a change count
    says "changed" on the run where the change deleted a step of the loop."""
    warnings = screen_optimizer_result(
        {"generate": "a", "ensure_correct": "b"}, {"generate": "x", "ensure_correct": "  "}
    )
    assert any("ensure_correct" in w and "empty" in w for w in warnings), warnings


@pytest.mark.parametrize("name", ["ensure_corect", "plan"])
def test_a_treatment_gene_the_genome_does_not_have_is_refused(name: str) -> None:
    """A typo, or a gene the genome never had, used to return no warning at all, which reads as
    "the screen looked at the treatment gene and found it moved". A screen that could not look
    must say so, so an unknown name is an error rather than an empty list."""
    before = {"generate": "a", "ensure_correct": "b"}
    after = {"generate": "x", "ensure_correct": "b"}
    with pytest.raises(ValueError, match=name):
        screen_optimizer_result(before, after, treatment_genes=[name])
    with pytest.raises(ValueError, match=name):
        screen_optimizer_result(before, before, treatment_genes=[name])
