"""LoopGEPA core data models.

Pydantic v2 models at every boundary, plus the deterministic confidence
assessment mandated by ADR-005 (confidence is computed, never guessed by an LLM).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

# Bump when any persisted record's shape changes (ADR-004.4). Stored on every
# trace/genome/variant so migration is always possible.
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


class Confidence(BaseModel):
    band: ConfidenceBand
    score: float


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
# imported sessions; the rest are only real inside the LoopGEPA runner (ADR-002).
_FILE_BACKED_GENE_TYPES = frozenset(
    {GeneType.SKILL, GeneType.AGENT, GeneType.PROMPT, GeneType.HOOK}
)


class LoopGene(BaseModel):
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


class LoopGenome(BaseModel):
    id: str
    name: str
    schema_version: str = SCHEMA_VERSION
    version: str = "0.1.0"
    description: str | None = None
    genes: list[LoopGene] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    # Required (never omitted) so a proposal always names its evidence — an empty
    # list is allowed but forces low confidence (ADR-002 / risk-register 17.2).
    evidence_trace_ids: list[str]
    target_gene_id: str | None = None
    parent_variant_id: str | None = None
    before: str | dict[str, Any] | None = None
    after: str | dict[str, Any] | None = None
    expected_effects: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: Confidence | None = None
    requires_human_approval: bool = True
