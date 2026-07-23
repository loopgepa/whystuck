"""Budget guard for the one paid validation-study run.

The study's headline safety property is a hard USD cap. This module is the rail:
a pricing table + a per-call-group plan + a `CostGuard` that pre-counts the
planned spend and refuses to proceed if it would breach the ceiling. Pure
arithmetic, no network — the guard runs *before* any live call.

What is a hard bound vs. a calibrated estimate (be precise — the guard is only as
honest as this distinction):
- **Call counts** are hard bounds. Worker calls account for GEPA's search, both
  held-out evals, AND `optimize_loop`'s post-search train-set re-scoring; reflection
  calls are bounded because the live run passes `reflection_calls` to GEPA as
  `max_candidate_proposals` (one reflection LM call per proposal). This module is
  pure arithmetic and can't enforce those itself, so the bounds hold iff the caller
  wires the caps through — `run_validation_study.py` does. (GEPA's own
  `max_reflection_cost` stopper is inert for a callable reflection_lm, which the
  cassette wrapper always supplies, so the *proposal* cap is what bounds reflection.)
- **Output tokens** are a hard bound: the live run enforces `max_tokens` on both the
  worker (`WORKER_MAX_OUTPUT_TOKENS`) and reflection (`REFLECTION_MAX_OUTPUT_TOKENS`)
  completions, so no call emits more than priced.
- **Input tokens** are a CONSERVATIVE ESTIMATE, not a hard bound. The worker input
  shape is sized for a realistic GEPA-evolved gene (see `WORKER_TOKENS`), but GEPA
  can in principle propose a longer instruction (up to `REFLECTION_MAX_OUTPUT_TOKENS`),
  and its reflection prompt size is GEPA-internal — neither is capped here. So the
  dollar estimate is conservative for typical runs but a pathological reflection that
  emits maximal-length instructions every round could exceed it. The guard checks the
  estimate pre-flight against `--max-usd`; it is not a runtime meter that aborts mid-run.

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
    # Cross-family critic options for run 6 (D1). gpt-5.6-luna is the OpenAI-only
    # cross-LINEAGE critic we run now; the other two are true cross-PROVIDER critics
    # for the stronger follow-up (each needs that provider's API key).
    "gemini/gemini-2.5-flash-lite": (0.10, 0.40),  # cheapest true cross-family critic
    "anthropic/claude-haiku-4-5": (1.00, 5.00),  # cross-family critic (Anthropic)
}


# Enforced hard caps on completion length (passed to litellm as `max_tokens`), so
# OUTPUT tokens — the dominant cost — are a genuine ceiling the guard can rely on,
# not a hopeful estimate. The live run MUST wire these through (run_validation_study
# builds its `complete`/`reflection` with exactly these caps) or the bound is void.
WORKER_MAX_OUTPUT_TOKENS = 320  # ~240 words; ample for IFBench constraint-following
# The v2 code study needs a larger worker cap: a solution is a whole function (imports +
# body), and the refine step re-emits it. 512 comfortably fits an MBPP+ function while
# still hard-capping output. Priced via CODE_WORKER_TOKENS below.
CODE_WORKER_MAX_OUTPUT_TOKENS = 512
REFLECTION_MAX_OUTPUT_TOKENS = 1500

# Per-call (in, out) shape for the worker. Output is the enforced hard cap. Input is
# a CONSERVATIVE ESTIMATE (see module docstring): the seed-gene loop prompts measured
# at 545 / 709 / 771 tokens (generate / critique / refine) over the held-out IFBench
# set, but GEPA rewrites the genes during arms B/D, and an evolved instruction embeds
# into every later worker prompt. 1200 covers the worst case with a gene of ~360
# tokens — comfortably above a typical GEPA-evolved instruction (a paragraph or two),
# and `refine` (the largest shape: one gene + prior draft ≤320 + critique ≤320 + task)
# stays under it. It is NOT an absolute bound: a pathological gene approaching the
# reflection output cap could exceed it. Output stays at the enforced cap; critique
# output is in practice far below it, so that dimension carries extra slack.
WORKER_TOKENS = (1200, WORKER_MAX_OUTPUT_TOKENS)  # (in estimate, out hard cap)
# Code worker input is larger than IFBench's: the prompt carries the task + the VISIBLE
# base tests, and the refine prompt adds the prior solution (≤512) + the critique (which
# embeds a base-test traceback, capped at ~300 chars) + an evolved gene. 2200 is a
# conservative upper bound for that worst-case (refine) shape; output stays hard-capped.
CODE_WORKER_TOKENS = (2200, CODE_WORKER_MAX_OUTPUT_TOKENS)  # (in estimate, out hard cap)
# Reflection output is enforced via max_tokens on the reflection LM; the input (GEPA's
# prompt over sampled traces) is a calibrated estimate — GEPA sizes that prompt itself,
# so it can't be hard-capped here. The proposal-count cap (max_candidate_proposals) is
# what makes reflection spend bounded regardless; this shape just prices each proposal.
REFLECTION_TOKENS = (4000, REFLECTION_MAX_OUTPUT_TOKENS)  # (in, out)

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
    worker_calls_per_task: int = 1,
    worker_tokens: tuple[int, int] = WORKER_TOKENS,
) -> list[CallGroup]:
    """Conservative per-call plan for a study configuration.

    Worker (LLM task) *task-evaluations*, per arm:
    - every arm scores the seed genome on held-out → `n_heldout` evals;
    - each LLM-optimizing arm (B, D) additionally spends ~`max_metric_calls`
      task-evals during GEPA search, `n_train` to re-score the winner on the
      train set (`optimize_loop`'s post-search objectives/frontier pass, which
      re-invokes the worker per train task), and `n_heldout` to score the
      winner on held-out;
    - arm C's optimization AND its post-search re-scoring use the *command*
      worker (free, not counted); it still scores seed + winner on held-out via
      the LLM worker → `2 * n_heldout`.

    `worker_calls_per_task` converts task-evals to LLM calls: the self-refine loop
    worker makes up to `1 + 2 * loop_max_retries` calls per eval (generate + each
    critique/refine round), so a single-shot worker passes 1 and the default loop
    (2 retries) passes 5. This is the fix that keeps the $5 guard honest once the
    worker is a multi-call loop rather than one shot.

    `n_train` is explicit (not assumed equal to `n_heldout`) so the train-set
    re-scoring term stays correct if the two splits diverge. `n_seeds` scales the
    whole plan — the study runs every arm once per seed for the mean ± std, and
    the guard MUST account for all of them or it under-counts. Reflection:
    `reflection_calls` per LLM-optimizing arm — a hard bound, because the caller
    wires it through as `max_candidate_proposals` (see the module docstring).

    `worker_tokens` is the per-worker-call (in, out) shape; it defaults to the IFBench
    `WORKER_TOKENS` and the caller passes `CODE_WORKER_TOKENS` for the (larger) code
    study, so the guard prices whichever task set actually runs.
    """
    worker_evals = 0
    for arm in arms:
        if arm in _LLM_WORKER_ARMS:
            worker_evals += n_heldout  # seed held-out eval
        if arm in _LLM_OPTIMIZING_ARMS:
            # GEPA search + post-search train re-scoring + winner held-out eval
            worker_evals += max_metric_calls + n_train + n_heldout
        elif arm == "C":
            worker_evals += n_heldout  # winner held-out eval (command opt is free)

    reflection_per_seed = reflection_calls * sum(1 for a in arms if a in _LLM_OPTIMIZING_ARMS)
    worker_calls = worker_evals * worker_calls_per_task * n_seeds
    reflection_total = reflection_per_seed * n_seeds

    return [
        CallGroup(worker_model, worker_calls, *worker_tokens),
        CallGroup(reflection_model, reflection_total, *REFLECTION_TOKENS),
    ]
