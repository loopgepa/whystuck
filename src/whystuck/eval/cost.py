"""Budget guard for the one paid validation-study run.

The study's headline safety property is a hard USD cap. This module is the rail:
a pricing table + a per-call-group plan + a `CostGuard` that pre-counts the
planned spend and refuses to proceed if it would breach the ceiling. Pure
arithmetic, no network — the guard runs *before* any live call.

Conservatism scope: the *worker*-call count is a genuine upper bound (it accounts
for GEPA's search, both held-out evals, AND `optimize_loop`'s post-search
train-set re-scoring). The *reflection*-call count is a caller-supplied estimate
(`reflection_calls`), not a proven bound — GEPA exposes no hard per-arm reflection
budget — so set it with margin and keep `--max-usd` comfortably above the estimate.

Prices are USD per 1M tokens (input, output), July 2026 OpenAI list prices.
Update the table when prices move; the estimator reads only from it, so a stale
entry can never silently under-count (an unknown model raises).
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens: model -> (input, output). Verified July 2026.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini": (0.80, 3.20),  # GEPA's exact IFBench task LM (spec §6)
    "openai/gpt-4.1-nano": (0.10, 0.40),  # cheaper worker option; not in the spec
    "openai/gpt-5.6-luna": (1.00, 6.00),  # extra reflection option beyond the spec
    "openai/gpt-5.6-terra": (2.50, 15.00),  # reflection default (spec §6)
    "openai/gpt-5.6-sol": (5.00, 30.00),  # reflection for the final published run (spec §6)
}


# Per-call token shapes, chosen to reproduce the design spec's §6 per-call cost
# approximations (worker c_w ≈ $0.0013, reflection c_r ≈ $0.033 with Terra).
WORKER_TOKENS = (600, 250)  # (in, out) — a short IFBench completion
REFLECTION_TOKENS = (4000, 1500)  # (in, out) — GEPA reflection over traces

# Which arms invoke the LLM worker / reflection model.
_LLM_OPTIMIZING_ARMS = frozenset({"B", "D"})  # worker opt + reflection
_LLM_WORKER_ARMS = frozenset({"A", "B", "C", "D"})  # all arms LLM-score held-out


class BudgetExceeded(Exception):
    """Raised when a planned run's estimated cost would breach the ceiling."""


@dataclass(frozen=True)
class CallGroup:
    """A batch of same-shaped model calls: `calls` calls to `model`, each with
    roughly `in_tokens` prompt tokens and `out_tokens` completion tokens."""

    model: str
    calls: int
    in_tokens: int
    out_tokens: int

    def usd(self) -> float:
        in_price, out_price = MODEL_PRICES[self.model]  # KeyError on unknown model, by design
        per_call = (self.in_tokens * in_price + self.out_tokens * out_price) / 1_000_000
        return self.calls * per_call


def estimate_usd(groups: list[CallGroup]) -> float:
    """Total estimated USD for a plan. Raises KeyError on an unpriced model."""
    return sum(g.usd() for g in groups)


@dataclass
class CostGuard:
    """Refuses a plan whose estimated cost exceeds `max_usd`."""

    max_usd: float

    def check(self, groups: list[CallGroup]) -> float:
        """Return the estimated USD if within budget; raise `BudgetExceeded` if not."""
        estimate = estimate_usd(groups)
        if estimate > self.max_usd:
            raise BudgetExceeded(
                f"planned run ≈ ${estimate:.4f} exceeds the ${self.max_usd} ceiling; "
                f"lower max_metric_calls / arms / task counts, or raise --max-usd"
            )
        return estimate


def plan_study_calls(
    *,
    arms: tuple[str, ...],
    n_train: int,
    n_heldout: int,
    max_metric_calls: int,
    reflection_calls: int,
    worker_model: str,
    reflection_model: str,
    n_seeds: int = 1,
) -> list[CallGroup]:
    """Conservative per-call plan for a study configuration.

    Worker (LLM task) calls, per arm:
    - every arm scores the seed genome on held-out → `n_heldout` calls;
    - each LLM-optimizing arm (B, D) additionally spends ~`max_metric_calls`
      worker calls during GEPA search, `n_train` to re-score the winner on the
      train set (`optimize_loop`'s post-search objectives/frontier pass, which
      re-invokes the LLM worker per train task), and `n_heldout` to score the
      winner on held-out;
    - arm C's optimization AND its post-search re-scoring use the *command*
      worker (free, not counted); it still scores seed + winner on held-out via
      the LLM worker → `2 * n_heldout`.

    `n_train` is explicit (not assumed equal to `n_heldout`) so the train-set
    re-scoring term stays correct if the two splits diverge. `n_seeds` scales the
    whole plan — the study runs every arm once per seed for the mean ± std, and
    the guard MUST account for all of them or it under-counts. Reflection: a
    caller estimate of `reflection_calls` per LLM-optimizing arm — see the module
    docstring on why that term is an estimate, not a proven bound.
    """
    worker_calls = 0
    for arm in arms:
        if arm in _LLM_WORKER_ARMS:
            worker_calls += n_heldout  # seed held-out eval
        if arm in _LLM_OPTIMIZING_ARMS:
            # GEPA search + post-search train re-scoring + winner held-out eval
            worker_calls += max_metric_calls + n_train + n_heldout
        elif arm == "C":
            worker_calls += n_heldout  # winner held-out eval (command opt is free)

    reflection_per_seed = reflection_calls * sum(1 for a in arms if a in _LLM_OPTIMIZING_ARMS)
    worker_calls *= n_seeds
    reflection_total = reflection_per_seed * n_seeds

    return [
        CallGroup(worker_model, worker_calls, *WORKER_TOKENS),
        CallGroup(reflection_model, reflection_total, *REFLECTION_TOKENS),
    ]
