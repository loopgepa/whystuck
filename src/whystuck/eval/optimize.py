"""Optimize a loop by wrapping the official GEPA optimizer (ADR-011).

GEPA owns the reflective search + Pareto frontier; WhyStuck supplies the
loop-genome adapter and scores the winner with its own deterministic objectives
(so the reported score never depends on GEPA's internals). `gepa` is imported
lazily — the base install and the whole v0.1 core stay free of it.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from whystuck.core.models import LoopVariant, ParetoCandidate
from whystuck.core.storage import WhyStuckStore
from whystuck.core.util import mean, utc_now_iso
from whystuck.eval.adapter import (
    LoopGenomeAdapter,
    RunTask,
    VerifyTruth,
    candidate_from_genome,
    genome_from_candidate,
)
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.objectives import objective_vector

OptimizeFn = Callable[..., dict[str, Any]]


def optimize_loop(
    store: WhyStuckStore,
    loop_id: str,
    tasks: list[BenchmarkTask],
    run_task: RunTask,
    *,
    reflection_lm: str | Callable[[str], str],
    max_metric_calls: int = 50,
    max_candidate_proposals: int | None = None,
    module_selector: str = "round_robin",
    verify_gene: str | None = None,
    verify_truth: VerifyTruth | None = None,
    _optimize: OptimizeFn | None = None,
) -> dict[str, Any]:
    """Run GEPA over a loop's text genes; save the winner as a variant + frontier.

    `max_metric_calls` bounds worker (task LM) calls; `max_candidate_proposals`,
    when set, bounds the number of GEPA proposals — one reflection LM call each —
    so the reflection spend is a *hard* cap, not an estimate (GEPA's
    `max_reflection_cost` stopper is inert for a callable reflection_lm, which is
    what the cassette wrapper always supplies). Left `None`, GEPA stops on
    `max_metric_calls` alone.

    `module_selector` decides how many components one proposal may rewrite, and it is
    load-bearing for arm fairness. GEPA's default `"round_robin"` selects exactly ONE
    component per proposal and cycles, so a two-gene arm spends half its proposals on
    each gene and searches the productive gene half as hard as a one-gene arm under the
    same cap — a budget-per-gene confound that a matched-total-budget comparison cannot
    see. `"all"` rewrites every component per proposal, which is the fair comparison for
    "does optimizing more of the loop help". The default stays `"round_robin"` because
    the committed cassettes were recorded under it: changing it changes every reflection
    prompt, hence every cassette key, and would invalidate the released replays.

    `verify_gene`/`verify_truth` enable component-specific reflection for the critique
    gene (see `LoopGenomeAdapter`). Both default to None, the released behaviour.
    """
    base = store.load_loop(loop_id)
    adapter = LoopGenomeAdapter(base, run_task, verify_gene=verify_gene, verify_truth=verify_truth)
    optimizer = _optimize or _gepa_optimize
    optimizer_kwargs: dict[str, Any] = {
        "seed_candidate": candidate_from_genome(base),
        "trainset": tasks,
        "adapter": adapter,
        "reflection_lm": reflection_lm,
        "max_metric_calls": max_metric_calls,
        "module_selector": module_selector,
    }
    if max_candidate_proposals is not None:
        optimizer_kwargs["max_candidate_proposals"] = max_candidate_proposals
    result = optimizer(**optimizer_kwargs)
    best_candidate = result["best_candidate"]

    # Score the winner ourselves — deterministic, GEPA-internals-independent.
    scored = adapter.evaluate(tasks, best_candidate, capture_traces=True)
    best_score = round(mean(scored.scores), 4)
    traj = scored.trajectories or []
    objectives = _average_objectives([objective_vector(t.metrics) for t in traj])

    variant = LoopVariant(
        id=f"variant-{uuid.uuid4().hex[:8]}",
        loop_id=loop_id,
        genome=genome_from_candidate(base, best_candidate),
        created_at=utc_now_iso(),
        tags=["optimized"],
    )
    store.save_variant(variant)
    frontier = [ParetoCandidate(variant_id=variant.id, score=best_score, objectives=objectives)]
    store.save_frontier(loop_id, frontier)
    return {
        "variant_id": variant.id,
        "best_score": best_score,
        "frontier": [c.model_dump() for c in frontier],
    }


def _average_objectives(vectors: list[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    keys = vectors[0].keys()
    return {k: round(mean([v[k] for v in vectors]), 4) for k in keys}


def _gepa_optimize(**kwargs: Any) -> dict[str, Any]:
    """The real GEPA call. Requires `pip install whystuck[eval]` and a network
    reflection LM; not exercised in the offline test suite."""
    try:
        import gepa
        from gepa.core.adapter import EvaluationBatch, GEPAAdapter
        from gepa.utils.stop_condition import MaxCandidateProposalsStopper
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "loop optimization requires the eval extra: pip install whystuck[eval]"
        ) from exc

    our = kwargs["adapter"]

    class _Bridge(GEPAAdapter):  # type: ignore[misc]
        def evaluate(self, batch, candidate, capture_traces=False):  # type: ignore[no-untyped-def]
            b = our.evaluate(batch, candidate, capture_traces=capture_traces)
            return EvaluationBatch(outputs=b.outputs, scores=b.scores, trajectories=b.trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):  # type: ignore[no-untyped-def]
            return our.make_reflective_dataset(candidate, eval_batch, components_to_update)

    # Bound reflection LM calls to a HARD cap. GEPA makes one reflection call per
    # candidate proposal, so a MaxCandidateProposalsStopper caps reflection spend.
    # (GEPA's max_reflection_cost stopper is inert for a callable reflection_lm —
    # it reports $0 cost — and gepa.optimize has no max_candidate_proposals kwarg,
    # so the stopper is the only reliable lever.) GEPA ORs this with its own
    # max_metric_calls stopper (CompositeStopper mode="any"), so whichever bound
    # fires first stops the run — both remain genuine ceilings.
    optional: dict[str, Any] = {}
    if kwargs.get("max_candidate_proposals") is not None:
        cap = kwargs["max_candidate_proposals"]
        optional["stop_callbacks"] = [MaxCandidateProposalsStopper(cap)]
    result = gepa.optimize(
        seed_candidate=kwargs["seed_candidate"],
        trainset=kwargs["trainset"],
        adapter=_Bridge(),
        reflection_lm=kwargs["reflection_lm"],
        max_metric_calls=kwargs["max_metric_calls"],
        # Passing this explicitly (rather than relying on GEPA's default) makes the
        # selector part of the recorded configuration, so a reader can tell whether a
        # multi-gene arm was budget-starved per gene. "round_robin" reproduces the
        # committed runs byte-for-byte; "all" is the fair-budget comparison.
        module_selector=kwargs.get("module_selector", "round_robin"),
        **optional,
    )
    return {"best_candidate": result.best_candidate}
