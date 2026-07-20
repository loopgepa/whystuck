"""Tests for loopgepa.core.models — behavior, not pydantic plumbing."""

import pytest
from pydantic import ValidationError

from loopgepa.core.models import (
    SCHEMA_VERSION,
    ConfidenceBand,
    GeneType,
    LoopGene,
    LoopTrace,
    MutationProposal,
    MutationType,
    assess_confidence,
)


class TestAssessConfidence:
    """ADR-005: confidence is a deterministic function of trace count and the
    dominant-failure-mode cluster share, never an LLM guess."""

    def test_few_traces_is_low_and_capped_even_when_cluster_is_total(self):
        # 3 traces all in one mode: strong cluster, but too few to trust.
        result = assess_confidence(n_traces=3, cluster_share=1.0)
        assert result.band is ConfidenceBand.LOW
        assert result.score <= 0.4

    def test_mid_traces_is_medium(self):
        result = assess_confidence(n_traces=10, cluster_share=0.8)
        assert result.band is ConfidenceBand.MEDIUM

    def test_many_traces_with_strong_cluster_is_high(self):
        result = assess_confidence(n_traces=20, cluster_share=0.6)
        assert result.band is ConfidenceBand.HIGH

    def test_many_traces_with_weak_cluster_is_only_medium(self):
        # Plenty of data, but failures are scattered — not a confident single mode.
        result = assess_confidence(n_traces=20, cluster_share=0.3)
        assert result.band is ConfidenceBand.MEDIUM

    def test_cluster_share_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            assess_confidence(n_traces=20, cluster_share=1.5)


class TestLoopGene:
    """ADR-002: only file-backed genes are promotable for imported sessions;
    runner-backed genes are only real inside the LoopGEPA runner."""

    @pytest.mark.parametrize(
        "gene_type",
        [GeneType.SKILL, GeneType.AGENT, GeneType.PROMPT, GeneType.HOOK],
    )
    def test_file_backed_gene_types(self, gene_type):
        gene = LoopGene(id="g1", type=gene_type, name="x", content="y")
        assert gene.file_backed is True

    @pytest.mark.parametrize(
        "gene_type",
        [
            GeneType.VERIFIER_RUBRIC,
            GeneType.RETRY_POLICY,
            GeneType.STOP_CONDITION,
            GeneType.CONTEXT_POLICY,
            GeneType.BUDGET_POLICY,
        ],
    )
    def test_runner_backed_gene_types_are_not_file_backed(self, gene_type):
        # content={} is valid for all these types (budget_policy requires a dict).
        gene = LoopGene(id="g1", type=gene_type, name="x", content={})
        assert gene.file_backed is False


class TestLoopTrace:
    def test_trace_stamps_current_schema_version(self):
        # ADR-004.4: every persisted record carries a schema version.
        trace = LoopTrace(id="t1", loop_id="bugfix", created_at="2026-07-20T00:00:00Z")
        assert trace.schema_version == SCHEMA_VERSION

    def test_trace_defaults_outcome_to_unknown(self):
        trace = LoopTrace(id="t1", loop_id="bugfix", created_at="2026-07-20T00:00:00Z")
        assert trace.final_outcome == "unknown"


class TestMutationProposal:
    """ADR-002/17.2: a proposal must always name its evidence, even if empty."""

    def _kwargs(self, **overrides):
        base = dict(
            id="m1",
            loop_id="bugfix",
            mutation_type=MutationType.EDIT_GENE,
            target_gene_id="verifier_rubric",
            title="Strengthen rubric",
            rationale="Repairs repeat the original mistake.",
            after="check harder",
        )
        base.update(overrides)
        return base

    def test_evidence_trace_ids_is_required(self):
        with pytest.raises(ValidationError):
            MutationProposal(**self._kwargs())

    def test_empty_evidence_is_allowed_explicitly(self):
        proposal = MutationProposal(**self._kwargs(evidence_trace_ids=[]))
        assert proposal.evidence_trace_ids == []

    def test_requires_human_approval_defaults_true(self):
        proposal = MutationProposal(**self._kwargs(evidence_trace_ids=["t1"]))
        assert proposal.requires_human_approval is True
