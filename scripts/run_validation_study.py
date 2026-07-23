#!/usr/bin/env python
"""Run (or dry-run) the GEPA validation study — the ONE paid entrypoint.

Safety model: this is the only place in the repo that spends tokens, and it will
not spend a cent without `--yes`. The default is a dry run that prints the
planned per-call breakdown and the estimated cost, checks it against `--max-usd`,
and exits — no API key required. `--yes` additionally requires OPENAI_API_KEY and
runs the requested arms live (`--arms`, default all four A-D), recording every LLM
call to `--record` (defaults to `study-cassette.json` in the working directory if
omitted) so the numbers and cassettes are reproducible offline afterwards.

    # dry run — safe, offline, no key:
    python scripts/run_validation_study.py

    # the real run — the worker is a generate→critique→refine loop, so each task
    # eval is up to 1 + 2*loop_max_retries LLM calls; the default (2 seeds, mmc 64,
    # 10 proposals, 16/16 tasks, 1 retry) ≈ $4.5 under the $5 cap. Output tokens are
    # hard-capped; the input estimate is conservative for typical GEPA instructions
    # (see eval/cost.py). Needs ≥2 seeds for a verdict (a single seed is always
    # INCONCLUSIVE). Deeper search costs more — the guard refuses over $5.
    OPENAI_API_KEY=sk-... python scripts/run_validation_study.py --yes \
        --record tests/fixtures/cassettes/study.json --out docs/validation-study.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from whystuck.eval.cost import (
    MODEL_PRICES,
    REFLECTION_MAX_OUTPUT_TOKENS,
    WORKER_MAX_OUTPUT_TOKENS,
    WORKER_TOKENS,
    BudgetExceeded,
    CallGroup,
    CostGuard,
    estimate_usd,
    plan_study_calls,
)

_VALID_ARMS = frozenset("ABCD")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run/dry-run the GEPA validation study.")
    p.add_argument("--max-usd", type=float, default=5.0, help="hard spend ceiling (default 5.0)")
    p.add_argument("--arms", default="ABCD", help="which arms to run (default ABCD)")
    p.add_argument(
        "--task-set",
        choices=("ifbench", "multi", "multi-hard"),
        default="ifbench",
        help="ifbench = vendored single-constraint slice; multi = composable "
        "multi-constraint tasks (easy, saturated at seed 0.97); multi-hard = harder "
        "constraints (exact count / forbidden word / max length) with real headroom",
    )
    p.add_argument("--n-train", type=int, default=16, help="train task count (default 16)")
    p.add_argument("--n-heldout", type=int, default=16, help="held-out task count (default 16)")
    p.add_argument(
        "--max-metric-calls", type=int, default=64, help="GEPA budget per arm (default 64)"
    )
    p.add_argument(
        "--loop-max-retries",
        type=int,
        default=1,
        help="generate→critique→refine rounds per task (worker makes up to 1+2N calls/eval)",
    )
    p.add_argument(
        "--plan-gene",
        action="store_true",
        help="complexity lever: add a third `plan` gene (plan→generate→critique→refine); "
        "arm D co-evolves it, adding one worker call/eval (2+2N total)",
    )
    p.add_argument(
        "--feedback-stats",
        action="store_true",
        help="feedback lever: inject a measured (non-oracle) text-stats readout into the "
        "critique step; no extra LLM calls (stats computed locally)",
    )
    p.add_argument(
        "--critique-model",
        default=None,
        help="cross-family critic (run 6): route the critique LLM call to this model "
        "instead of the worker. Cross-lineage (e.g. gpt-5.6 over a gpt-4.1 worker) if "
        "same provider; a true cross-provider critic needs that provider's key",
    )
    p.add_argument(
        "--oracle-critique",
        action="store_true",
        help="ceiling lever (run 7): the critique gate uses the true metric, not an LLM "
        "self-check; no critique LLM call (1+N calls/eval). Deliberate oracle condition",
    )
    p.add_argument(
        "--reflection-calls",
        type=int,
        default=10,
        help="hard cap on GEPA proposals per arm (= max reflection LM calls/arm)",
    )
    p.add_argument("--worker-model", default="openai/gpt-4.1-mini")
    p.add_argument("--reflection-model", default="openai/gpt-5.6-terra")
    p.add_argument(
        "--seeds", default="0,1", help="comma-separated seeds for mean±std (default 0,1)"
    )
    p.add_argument("--record", default=None, help="cassette JSON path to record calls into")
    p.add_argument("--out", default=None, help="where to write the results JSON")
    p.add_argument(
        "--report",
        default="docs/validation-study.md",
        help="where to write the Markdown report (default docs/validation-study.md)",
    )
    p.add_argument("--yes", action="store_true", help="actually spend and run live")
    return p.parse_args(argv)


def _seeds(args: argparse.Namespace) -> list[int]:
    return [int(s) for s in args.seeds.split(",") if s.strip() != ""]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    arms = tuple(args.arms)

    # Validate config at the cheap dry-run step, not mid-spend. Unknown arm
    # letters would otherwise plan to $0 (silent footgun) and only KeyError later.
    bad_arms = sorted(set(arms) - _VALID_ARMS)
    if bad_arms:
        print(
            f"ERROR: unknown arm(s) {bad_arms}; valid arms are A, B, C, D "
            f"(got --arms {args.arms!r}).",
            file=sys.stderr,
        )
        return 2
    model_args = [
        ("--worker-model", args.worker_model),
        ("--reflection-model", args.reflection_model),
    ]
    if args.critique_model:
        model_args.append(("--critique-model", args.critique_model))
    for label, model in model_args:
        if model not in MODEL_PRICES:
            print(
                f"ERROR: {label} {model!r} has no price in MODEL_PRICES; "
                f"known models: {sorted(MODEL_PRICES)}.",
                file=sys.stderr,
            )
            return 2

    seeds = _seeds(args)
    if not seeds:
        print("ERROR: --seeds must list at least one integer.", file=sys.stderr)
        return 2

    # Call count per eval: generate (+plan) + per retry a refine, plus a critique
    # UNLESS the oracle gate replaces it (no critique LLM call then).
    per_retry_calls = 1 if args.oracle_critique else 2
    worker_calls_per_task = (2 if args.plan_gene else 1) + per_retry_calls * args.loop_max_retries
    # Conservative pricing: a cross-family critic runs some worker-side calls on a
    # different (often pricier) model. We price ALL worker-side calls at the pricier
    # of {worker, critic} so the guard's estimate stays an upper bound (the true
    # split is cheaper). Irrelevant under --oracle-critique (no critique call).
    pricing_worker_model = args.worker_model
    if args.critique_model and not args.oracle_critique:
        # Pick the pricier model by TRUE cost at the real worker call shape (input-heavy),
        # not by unweighted sum(in,out) — a future entry with an unusual in/out ratio
        # could flip a sum-based comparison and silently under-price the guard.
        cost_at_shape = lambda m: CallGroup(m, 1, *WORKER_TOKENS).usd()  # noqa: E731
        if cost_at_shape(args.critique_model) > cost_at_shape(args.worker_model):
            pricing_worker_model = args.critique_model

    groups = plan_study_calls(
        arms=arms,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        max_metric_calls=args.max_metric_calls,
        reflection_calls=args.reflection_calls,
        worker_model=pricing_worker_model,
        reflection_model=args.reflection_model,
        n_seeds=len(seeds),
        worker_calls_per_task=worker_calls_per_task,
    )

    print(
        f"Planned study: arms={''.join(arms)}  seeds={seeds}  "
        f"max_metric_calls={args.max_metric_calls}"
    )
    for g in groups:
        print(f"  {g.model:24} {g.calls:5d} calls  ≈ ${g.usd():.4f}")
    estimate = estimate_usd(groups)
    print(f"Estimated total: ${estimate:.4f}   (ceiling ${args.max_usd})")

    try:
        CostGuard(max_usd=args.max_usd).check(groups)
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if not args.yes:
        print("\nDry run only. Re-run with --yes (and OPENAI_API_KEY set) to spend.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: --yes given but OPENAI_API_KEY is not set.", file=sys.stderr)
        return 3

    return _run_live(args, arms, seeds, estimate)


def _run_live(
    args: argparse.Namespace, arms: tuple[str, ...], seeds: list[int], estimate: float
) -> int:
    """The live run (reached only with --yes + a key). Not exercised offline."""
    import tempfile
    from pathlib import Path

    from whystuck.core.storage import WhyStuckStore
    from whystuck.core.util import utc_now_iso
    from whystuck.eval.cassette import Cassette
    from whystuck.eval.ifbench import ifbench_feedback, ifbench_score, load_ifbench
    from whystuck.eval.llm_worker import litellm_complete
    from whystuck.eval.multi_constraint import load_multi_constraint, mc_feedback, mc_score
    from whystuck.eval.report import render_report
    from whystuck.eval.study import run_study
    from whystuck.eval.text_stats import text_stats

    # Task set selects the loader + the scorer/feedback the worker loop grades with.
    if args.task_set in ("multi", "multi-hard"):
        hard = args.task_set == "multi-hard"
        # Size the generated pool so each disjoint half comfortably covers the larger
        # of n_train/n_heldout — otherwise load_multi_constraint raises rather than
        # silently truncating below the requested (and cost-priced) sample size.
        mc_pool = max(64, 2 * max(args.n_train, args.n_heldout) + 2)

        def load_split(split: str, seed: int, n: int):
            return load_multi_constraint(split, seed=seed, n=n, pool=mc_pool, hard=hard)  # type: ignore[arg-type]

        score_fn, feedback_fn = mc_score, mc_feedback
        dataset = f"multi-constraint{'-hard' if hard else ''} (IFEval-style)"
    else:

        def load_split(split: str, seed: int, n: int):
            return load_ifbench(split=split, seed=seed, n=n)  # type: ignore[arg-type]

        score_fn, feedback_fn, dataset = ifbench_score, ifbench_feedback, "IFBench"

    record_path = args.record or "study-cassette.json"
    runs: list[dict[str, dict]] = []
    # Enforce the exact output caps the budget guard priced (cost.py), so the live
    # spend cannot exceed the pre-flight estimate on the dominant (output) axis.
    with Cassette(record_path, mode="record") as cassette:
        complete = cassette.wrap(
            litellm_complete(args.worker_model, max_tokens=WORKER_MAX_OUTPUT_TOKENS), name="worker"
        )
        reflection = cassette.wrap(
            litellm_complete(args.reflection_model, max_tokens=REFLECTION_MAX_OUTPUT_TOKENS),
            name="reflection",
        )
        # Cross-family critic (run 6): a separate model for the critique step only.
        # Skipped under --oracle-critique (the gate makes no critique LLM call).
        critique_complete = None
        if args.critique_model and not args.oracle_critique:
            critique_complete = cassette.wrap(
                litellm_complete(args.critique_model, max_tokens=WORKER_MAX_OUTPUT_TOKENS),
                name="critic",
            )
        for seed in seeds:
            train = load_split("train", seed, args.n_train)
            heldout = load_split("heldout", seed, args.n_heldout)
            with tempfile.TemporaryDirectory() as d:
                store = WhyStuckStore(Path(d))
                results = run_study(
                    store,
                    complete=complete,
                    reflection_lm=reflection,
                    train=train,
                    heldout=heldout,
                    arms=arms,
                    max_metric_calls=args.max_metric_calls,
                    max_candidate_proposals=args.reflection_calls,
                    loop_max_retries=args.loop_max_retries,
                    score_fn=score_fn,
                    feedback_fn=feedback_fn,
                    use_plan_gene=args.plan_gene,
                    stats_fn=text_stats if args.feedback_stats else None,
                    critique_complete=critique_complete,
                    oracle_critique=args.oracle_critique,
                )
            runs.append({name: vars(r) for name, r in results.items()})

    meta = {
        "dataset": dataset,
        "n_train": args.n_train,
        "n_heldout": args.n_heldout,
        "worker_model": args.worker_model,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        "loop_max_retries": args.loop_max_retries,
        "plan_gene": args.plan_gene,
        "feedback_stats": args.feedback_stats,
        "critique_model": args.critique_model if not args.oracle_critique else None,
        "oracle_critique": args.oracle_critique,
        "worker_max_output_tokens": WORKER_MAX_OUTPUT_TOKENS,
        "seeds": seeds,
        "estimated_usd": estimate,
        "generated": utc_now_iso(),
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps({"meta": meta, "runs": runs}, indent=2), encoding="utf-8"
        )
    if args.report:
        Path(args.report).write_text(render_report(runs, meta=meta), encoding="utf-8")
        print(f"Wrote report to {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
