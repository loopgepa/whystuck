"""WhyStuck core data models.

Pydantic v2 models at every boundary, plus the deterministic confidence
assessment mandated by ADR-005 (confidence is computed, never guessed by an LLM).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# Bump when any persisted record's shape changes (ADR-004.4). Stored on every
# persisted record (trace, genome, variant, promotion) so migration is possible.
SCHEMA_VERSION = "0.1"

# --- Confidence thresholds (ADR-005) ------------------------------------------
# Fewer than this many traces can never be more than LOW confidence: a strong
# cluster over 3 runs is still noise.
_MIN_TRACES_FOR_MEDIUM = 5
# At/above this trace count, a dominant failure mode can reach HIGH confidence.
_MIN_TRACES_FOR_HIGH = 15
# Share of failing traces in the single dominant mode needed for HIGH: below
# this the failures are too scattered to attribute confidently.
_STRONG_CLUSTER_SHARE = 0.5

# Score ceilings per band — a band's numeric score never exceeds its cap, so
# "few traces" stays quantitatively low even when the cluster share is total.
_LOW_SCORE_CAP = 0.4
_MEDIUM_SCORE_CAP = 0.7
_HIGH_SCORE_CAP = 1.0


class ConfidenceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_BAND_CAP: dict[ConfidenceBand, float] = {
    ConfidenceBand.LOW: _LOW_SCORE_CAP,
    ConfidenceBand.MEDIUM: _MEDIUM_SCORE_CAP,
    ConfidenceBand.HIGH: _HIGH_SCORE_CAP,
}


class Confidence(BaseModel):
    band: ConfidenceBand
    score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _score_within_band_cap(self) -> Confidence:
        cap = _BAND_CAP[self.band]
        if self.score > cap:
            raise ValueError(f"score {self.score} exceeds {self.band} cap {cap}")
        return self


def assess_confidence(n_traces: int, cluster_share: float) -> Confidence:
    """Deterministically map (trace count, dominant-cluster share) to confidence.

    `cluster_share` is the fraction of failing traces attributed to the single
    dominant failure mode, in [0, 1].
    """
    if not 0.0 <= cluster_share <= 1.0:
        raise ValueError(f"cluster_share must be in [0, 1], got {cluster_share}")

    if n_traces < _MIN_TRACES_FOR_MEDIUM:
        band, cap = ConfidenceBand.LOW, _LOW_SCORE_CAP
    elif n_traces >= _MIN_TRACES_FOR_HIGH and cluster_share >= _STRONG_CLUSTER_SHARE:
        band, cap = ConfidenceBand.HIGH, _HIGH_SCORE_CAP
    else:
        band, cap = ConfidenceBand.MEDIUM, _MEDIUM_SCORE_CAP

    return Confidence(band=band, score=round(min(cap, cluster_share), 2))


# --- Genome -------------------------------------------------------------------


class GeneType(StrEnum):
    SKILL = "skill"
    AGENT = "agent"
    PROMPT = "prompt"
    HOOK = "hook"
    VERIFIER_RUBRIC = "verifier_rubric"
    RETRY_POLICY = "retry_policy"
    CONTEXT_POLICY = "context_policy"
    TOOL_POLICY = "tool_policy"
    STOP_CONDITION = "stop_condition"
    ESCALATION_POLICY = "escalation_policy"
    BUDGET_POLICY = "budget_policy"


# Genes backed by a file on disk (skill/agent/prompt/hook) can be promoted for
# imported sessions; the rest are only real inside the WhyStuck runner (ADR-002).
_FILE_BACKED_GENE_TYPES = frozenset(
    {GeneType.SKILL, GeneType.AGENT, GeneType.PROMPT, GeneType.HOOK}
)


class LoopGene(BaseModel):
    # validate_assignment so apply-time content changes re-check the invariant below.
    model_config = ConfigDict(validate_assignment=True)

    id: str
    type: GeneType
    name: str
    content: str | dict[str, Any]
    path: str | None = None
    evolvable: bool = True
    safety_locked: bool = False
    description: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_backed(self) -> bool:
        return self.type in _FILE_BACKED_GENE_TYPES

    @model_validator(mode="after")
    def _budget_content_is_dict(self) -> LoopGene:
        # A budget_policy gene must carry structured (dict) content so the budget
        # guard can compare caps; a str would let the guard silently no-op (H1).
        if self.type is GeneType.BUDGET_POLICY and not isinstance(self.content, dict):
            raise ValueError("budget_policy gene content must be a dict")
        return self


class LoopGenome(BaseModel):
    id: str
    name: str
    schema_version: str = SCHEMA_VERSION
    version: str = "0.1.0"
    description: str | None = None
    genes: list[LoopGene] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_gene_ids(self) -> LoopGenome:
        # Duplicate ids would make later mutations resolve first-match silently and
        # break attribution/lineage (ADR-002).
        ids = [g.id for g in self.genes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate gene ids in genome")
        return self


# --- Trace --------------------------------------------------------------------


class StepStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class ToolCall(BaseModel):
    name: str
    input_summary: str | None = None
    output_summary: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None


class LoopStepTrace(BaseModel):
    step_id: str
    role: str | None = None
    status: StepStatus = StepStatus.UNKNOWN
    input_summary: str | None = None
    output_summary: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    duration_ms: int | None = None


class LoopMetrics(BaseModel):
    success: bool | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    retry_count: int = 0
    human_interventions: int = 0
    changed_files_count: int | None = None
    diff_size_lines: int | None = None
    tests_passed: bool | None = None
    safety_score: float | None = None


class LoopTrace(BaseModel):
    id: str
    loop_id: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    variant_id: str | None = None
    task_id: str | None = None
    task_summary: str | None = None
    steps: list[LoopStepTrace] = Field(default_factory=list)
    metrics: LoopMetrics = Field(default_factory=LoopMetrics)
    final_outcome: Literal["success", "failure", "unknown"] = "unknown"
    failure_summary: str | None = None
    human_feedback: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Mutation -----------------------------------------------------------------


class MutationType(StrEnum):
    EDIT_GENE = "edit_gene"
    ADD_GENE = "add_gene"
    REMOVE_GENE = "remove_gene"
    LOCK_GENE = "lock_gene"


class MutationProposal(BaseModel):
    id: str
    loop_id: str
    mutation_type: MutationType
    title: str
    rationale: str
    # Required (never omitted) so a proposal always names its evidence. An empty
    # list is allowed; the server recomputes confidence from the evidence and
    # forces LOW when it is empty (see tools.propose_mutations) — this field does
    # not enforce that itself (IMPLEMENTATION §13, ADR-005).
    evidence_trace_ids: list[str]
    target_gene_id: str | None = None
    parent_variant_id: str | None = None
    before: str | dict[str, Any] | None = None
    after: str | dict[str, Any] | None = None
    expected_effects: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    requires_human_approval: bool = True

    @model_validator(mode="after")
    def _validate_shape(self) -> MutationProposal:
        # Field validity depends on mutation_type; fail at construction, not apply.
        if self.mutation_type is MutationType.ADD_GENE:
            if not isinstance(self.after, dict):
                raise ValueError("ADD_GENE requires `after` to be a gene definition dict")
        else:
            if self.target_gene_id is None:
                raise ValueError(f"{self.mutation_type} requires target_gene_id")
            if self.mutation_type is MutationType.EDIT_GENE and self.after is None:
                raise ValueError("EDIT_GENE requires `after`")
        return self


# --- Variant & promotion ------------------------------------------------------


class LoopVariant(BaseModel):
    id: str
    loop_id: str
    genome: LoopGenome
    created_at: str
    schema_version: str = SCHEMA_VERSION
    parent_variant_id: str | None = None
    mutations: list[MutationProposal] = Field(default_factory=list)
    metrics: LoopMetrics | None = None
    tags: list[str] = Field(default_factory=list)
    promoted: bool = False


class PromotionRecord(BaseModel):
    timestamp: str
    loop_id: str
    to_variant_id: str
    schema_version: str = SCHEMA_VERSION
    from_variant_id: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    approved_by: str | None = None
