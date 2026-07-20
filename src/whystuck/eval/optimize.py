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
    reflection_lm: str,
    max_metric_calls: int = 50,
    _optimize: OptimizeFn | None = None,
) -> dict[str, Any]:
    """Run GEPA over a loop's text genes; save the winner as a variant + frontier."""
    base = store.load_loop(loop_id)
    adapter = LoopGenomeAdapter(base, run_task)
    optimizer = _optimize or _gepa_optimize
    result = optimizer(
        seed_candidate=candidate_from_genome(base),
        trainset=tasks,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
    )
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
        import gepa  # type: ignore[import-not-found]
        from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # type: ignore[import-not-found]
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

    result = gepa.optimize(
        seed_candidate=kwargs["seed_candidate"],
        trainset=kwargs["trainset"],
        adapter=_Bridge(),
        reflection_lm=kwargs["reflection_lm"],
        max_metric_calls=kwargs["max_metric_calls"],
    )
    return {"best_candidate": result.best_candidate}
