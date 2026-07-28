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
    # 10 proposals, 16/16 tasks, 1 retry) ≈ $2.7 under the $5 cap. Output tokens are
    # hard-capped; the input estimate is conservative for typical GEPA instructions
    # (see eval/cost.py). Needs ≥2 seeds for a verdict (a single seed is always
    # INCONCLUSIVE). Deeper search costs more — the guard refuses over $5.
    OPENAI_API_KEY=sk-... python scripts/run_validation_study.py --yes \
        --record tests/fixtures/cassettes/study.json --out docs/validation-study.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from whystuck.eval.cost import (
    BCB_WORKER_MAX_OUTPUT_TOKENS,
    BCB_WORKER_TOKENS,
    CODE_WORKER_MAX_OUTPUT_TOKENS,
    CODE_WORKER_TOKENS,
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
        choices=("ifbench", "multi", "multi-hard", "code", "bcb"),
        default="ifbench",
        help="ifbench = vendored single-constraint slice; multi = composable "
        "multi-constraint tasks (easy, saturated at seed 0.97); multi-hard = harder "
        "constraints with real headroom; code = MBPP+ (base VISIBLE / plus HIDDEN) — "
        "SATURATES a capable worker (see the appendix); bcb = BigCodeBench-Hard in a "
        "Docker sandbox (visible/hidden unittest-method split), hard even for frontier "
        "models — the headroom-bearing v2 code study",
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
        "--actuator",
        action="store_true",
        help="actuator experiment (bcb only): add an `actuator` gene that turns the test "
        "signal + visible-test source into the refine step's repair guidance; arm D evolves "
        "generate+actuator with the detector held fixed. Adds one worker call/retry (3N).",
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
        "--feedback-level",
        choices=("lo", "mid", "hi"),
        default="mid",
        help="feedback-actionability lever (D2', ONLY with --oracle-critique): shape the "
        "repair text fed to the refine step. lo=verdict only; mid=which constraints failed "
        "(default, identical to run 8); hi=concrete measured edits (multi task sets only). "
        "Same call count as mid, so the cost estimate is unchanged",
    )
    p.add_argument(
        "--reflection-calls",
        type=int,
        default=10,
        help="hard cap on GEPA proposals per arm (= max reflection LM calls/arm)",
    )
    p.add_argument(
        "--module-selector",
        choices=("all", "round_robin"),
        default="all",
        help=(
            "how many genes one GEPA proposal may rewrite. 'round_robin' (GEPA's own "
            "default, and what every committed run used) rewrites ONE gene per proposal "
            "and cycles, so a multi-gene arm searches each gene only 1/k as hard as a "
            "one-gene arm under the same cap — a budget-per-gene confound. 'all' rewrites "
            "every gene per proposal, the fair loop-level-vs-prompt-only comparison. "
            "Changing this changes every reflection prompt, so runs recorded under one "
            "value cannot replay under the other."
        ),
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
    p.add_argument(
        "--replay",
        action="store_true",
        help="reproduce a prior run OFFLINE from its --record cassette: no API key, no "
        "spend, and any cassette miss raises (strict). Re-runs the full pipeline (incl. "
        "GEPA) against cached calls and re-writes --out/--report for bit-for-bit checking",
    )
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

    # Feedback-actionability lever validation, at the cheap dry-run step so a
    # misconfigured level can never reach a paid run. lo/hi are inert off the oracle
    # path (there the LLM writes the critique), and hi's edit directives are
    # multi_constraint-specific — either misconfig would silently waste money.
    # The lever acts either on the ORACLE path (multi/multi-hard, where the LLM critique is
    # replaced) or on the bcb EXEC path (where it shapes the executable visible-test feedback).
    # Off both — e.g. plain ifbench, or code (whose code_feedback has no level) — lo/hi are inert.
    _exec_lever = args.task_set == "bcb"
    if args.feedback_level != "mid" and not (args.oracle_critique or _exec_lever):
        print(
            f"ERROR: --feedback-level {args.feedback_level} only applies under --oracle-critique "
            "(the oracle path) or --task-set bcb (where it shapes the executable visible-test "
            "feedback). Off both paths the level is inert. Adjust the task set/oracle or drop "
            "--feedback-level.",
            file=sys.stderr,
        )
        return 2
    # The executable code task sets' anti-oracle protocol REQUIRES the non-oracle,
    # visible-test-gated critique: --oracle-critique gates on score_fn (= the HIDDEN tests),
    # which would leak the held-out scorer into the loop. Refuse at the cheap dry-run step.
    if args.task_set in ("code", "bcb") and args.oracle_critique:
        print(
            f"ERROR: --oracle-critique is incompatible with --task-set {args.task_set}: it "
            "would gate the loop on the hidden test scorer (oracle contamination). These task "
            "sets use executable VISIBLE-test feedback in a non-oracle critique instead.",
            file=sys.stderr,
        )
        return 2

    # The detector's truth signal must be free of the held-out scorer AND cheap. On text
    # that is the programmatic verifier (pure Python, no execution). On the code task sets
    # score_fn runs the HIDDEN suite, so wiring it as detector truth would both put the
    # held-out signal into the optimizer's channel and pay real container execution on
    # every critique step. Refuse rather than silently doing the wrong one.

    if args.feedback_level == "hi" and args.task_set not in ("multi", "multi-hard", "bcb"):
        print(
            "ERROR: --feedback-level hi needs a multi task set (--task-set multi|multi-hard), "
            "whose concrete edit directives are multi_constraint-specific, or --task-set bcb, "
            "where hi appends the visible-test source.",
            file=sys.stderr,
        )
        return 2

    if args.actuator and args.task_set != "bcb":
        print(
            "ERROR: --actuator needs --task-set bcb: its actuator context is the bcb "
            "visible-test source (bcb_feedback level=hi). Other task sets have no such context.",
            file=sys.stderr,
        )
        return 2
    if args.actuator and args.plan_gene:
        print(
            "ERROR: --actuator and --plan-gene are distinct experiments; enable at most one.",
            file=sys.stderr,
        )
        return 2

    # Call count per eval: generate (+plan) + per retry a refine, plus a critique
    # UNLESS the oracle gate replaces it (no critique LLM call then), plus an actuator
    # call per retry when the actuator gene is on (critique + actuator + refine = 3).
    if args.oracle_critique:
        per_retry_calls = 1
    elif args.actuator:
        per_retry_calls = 3
    else:
        per_retry_calls = 2
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

    # Code solutions are longer than IFBench text on both axes, so the guard must price
    # the code worker with its larger (in, out) shape or it under-counts a paid run.
    worker_tokens = {"code": CODE_WORKER_TOKENS, "bcb": BCB_WORKER_TOKENS}.get(
        args.task_set, WORKER_TOKENS
    )

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
        worker_tokens=worker_tokens,
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

    # Pre-spend guard for the code arm: actually EXECUTE the vendored references through
    # the sandbox on THIS host (the generic dry-run never runs code). Proves numpy imports
    # under the limits, the executor works here, and the data file is intact — so a broken
    # host/data crashes NOW (offline, no key) instead of silently producing all-zero
    # held-out scores after real spend. Runs on the dry-run path too, by design.
    if args.task_set == "code":
        from whystuck.eval.code_bench import load_code_tasks, preflight_code_tasks

        code_tasks = load_code_tasks("all", seed=0)
        print(f"Preflighting code sandbox + data ({len(code_tasks)} vendored references)...")
        preflight_code_tasks(code_tasks)
        print("  preflight OK: executor works here and every reference passes both suites.")
    elif args.task_set == "bcb":
        from whystuck.eval.bcb_bench import load_bcb_tasks, preflight_bcb_tasks

        bcb_tasks = load_bcb_tasks("all", seed=0)
        print(f"Preflighting BigCodeBench Docker sandbox + data ({len(bcb_tasks)} references)...")
        preflight_bcb_tasks(bcb_tasks)
        print("  preflight OK: Docker executor works here and every reference passes both splits.")

    # Replay reproduces a prior run from its cassette: offline, no key, no spend
    # (strict — a cassette miss raises). Takes precedence over --yes.
    if args.replay:
        return _run_live(args, arms, seeds, estimate)

    if not args.yes:
        print("\nDry run only. Re-run with --yes (and OPENAI_API_KEY set) to spend.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: --yes given but OPENAI_API_KEY is not set.", file=sys.stderr)
        return 3

    return _run_live(args, arms, seeds, estimate)


def _build_oracle_feedback(level: str, feedback_fn):
    """Build the oracle-path repair-feedback fn for a --feedback-level (D2' lever).

    Returns None for "mid" (the loop then falls back to `feedback_fn`, byte-identical
    to run 8). "lo" returns a fn that ignores the draft and emits only a bare verdict;
    "hi" returns a fn that appends `mc_edit_directives`' concrete measured edits after
    `feedback_fn`'s which-constraint prose. Hoisted to module scope (not an inline
    closure) so the exact production strings are unit-testable without a paid run.
    This function does NOT enforce oracle-critique-only / multi-task-set use — the CLI
    validation in `main()` does; a mis-level here would otherwise be invisible to tests.
    """

    def _lo(_output, _task):  # verdict only, no which-constraint
        return "not all requirements are satisfied; revise"

    def _hi(output, task):  # which-constraint + concrete measured edits
        from whystuck.eval.multi_constraint import mc_edit_directives

        base = feedback_fn(output, task)
        return f"{base}\n\nConcrete edits:\n{mc_edit_directives(output, task)}"

    return {"lo": _lo, "hi": _hi}.get(level)  # "mid" (and anything else) -> None


def _run_live(
    args: argparse.Namespace, arms: tuple[str, ...], seeds: list[int], estimate: float
) -> int:
    """The live run (reached only with --yes + a key). Not exercised offline."""
    import tempfile
    from pathlib import Path

    from whystuck.core.storage import WhyStuckStore
    from whystuck.core.util import utc_now_iso
    from whystuck.eval.bcb_bench import bcb_feedback, bcb_score, load_bcb_tasks
    from whystuck.eval.cassette import Cassette
    from whystuck.eval.code_bench import code_feedback, code_score, load_code_tasks
    from whystuck.eval.ifbench import ifbench_feedback, ifbench_score, load_ifbench
    from whystuck.eval.llm_worker import Feedback, litellm_complete
    from whystuck.eval.multi_constraint import load_multi_constraint, mc_feedback, mc_score
    from whystuck.eval.report import render_report
    from whystuck.eval.study import run_study
    from whystuck.eval.text_stats import text_stats

    # Task set selects the loader + the scorer/feedback the worker loop grades with.
    # `exec_feedback` is the task-aware, EXECUTABLE in-loop signal (code only); the text
    # task sets have no executor, so it stays None (their critique is LLM/oracle-gated).
    exec_feedback: Feedback | None = None
    # Actuator experiment (bcb): the actuator gene's raw materials = the visible-test result
    # PLUS the visible-test source (bcb_feedback level=hi), the same signal the hand-built +0.20
    # 'hi' lift used — so the experiment asks whether GEPA can evolve to exploit it.
    actuator_context: Feedback | None = None
    if args.task_set == "bcb":
        # BigCodeBench-Hard in Docker: visible unittest methods run in-loop (exec_feedback),
        # hidden methods score (held-out). Hard even for frontier models → real gradient.
        def load_split(split: str, seed: int, n: int):
            return load_bcb_tasks(split, seed=seed, n=n)  # type: ignore[arg-type]

        score_fn, feedback_fn = bcb_score, bcb_feedback
        # Feedback-actionability lever (code D2'): shape the in-loop visible-test signal at the
        # chosen level. mid (default) is byte-identical to the recorded runs; lo/hi are the
        # cross-domain analogue of the IFBench actionability ablation.
        exec_feedback = functools.partial(bcb_feedback, level=args.feedback_level)
        if args.actuator:
            actuator_context = functools.partial(bcb_feedback, level="hi")
        dataset = "BigCodeBench-Hard (Docker; visible methods in-loop, hidden methods scored)"
    elif args.task_set == "code":
        # MBPP+: base tests VISIBLE (shown in-prompt + executed by the in-loop critique
        # via exec_feedback), plus tests HIDDEN (the held-out scorer). See code_bench.
        def load_split(split: str, seed: int, n: int):
            return load_code_tasks(split, seed=seed, n=n)  # type: ignore[arg-type]

        score_fn, feedback_fn = code_score, code_feedback
        exec_feedback = code_feedback  # base-test execution read into the critique
        dataset = "MBPP+ (EvalPlus; base=visible/in-loop, plus=hidden/scored)"
    elif args.task_set in ("multi", "multi-hard"):
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

    # Detector truth for the verify gene's own reflection records — arm D's actual
    # treatment. Without it GEPA rewrites the critique gene while being shown the
    # GENERATOR's output, and duly proposes generation prose into a critique slot, so the
    # loop-level arm never receives a meaningful intervention.
    #
    # Built from the same programmatic verifier the study scores with, but consulted ONLY
    # when building reflection records on the TRAIN split: never inside the loop and never
    # at held-out scoring. That makes it the standard optimizer signal (which already
    # carries per-task feedback for the generate gene) made component-appropriate, not
    # oracle contamination of the measurement.
    #
    # Text task sets only. On `code`/`bcb` the equivalent truth is the VISIBLE test result;
    # `score_fn` there runs the HIDDEN suite, so wiring it here would both put the held-out
    # signal into the optimizer's channel and pay container execution per critique step.
    # Those families therefore keep component-agnostic reflection until a visible-test
    # verdict is threaded through, and the run metadata records which applied.
    verify_truth: Callable[[str, Any], tuple[bool, str]] | None = None
    if args.task_set not in ("code", "bcb"):

        def _verify_truth(output: str, task: Any) -> tuple[bool, str]:
            return score_fn(output, task) >= 1.0, feedback_fn(output, task)

        verify_truth = _verify_truth

    # Worker output hard-cap matches what the guard priced for this task set.
    worker_max_out = {
        "code": CODE_WORKER_MAX_OUTPUT_TOKENS,
        "bcb": BCB_WORKER_MAX_OUTPUT_TOKENS,
    }.get(args.task_set, WORKER_MAX_OUTPUT_TOKENS)

    # Feedback-actionability lever (D2'): the oracle-path repair-feedback fn for the
    # chosen level. None = "mid" (loop falls back to feedback_fn, byte-identical to
    # run 8). Only consulted under --oracle-critique (lo/hi require it; the CLI guards
    # above enforce that and that hi implies a multi task set).
    oracle_feedback = (
        _build_oracle_feedback(args.feedback_level, feedback_fn) if args.oracle_critique else None
    )

    record_path = args.record or "study-cassette.json"
    runs: list[dict[str, dict]] = []
    # Enforce the exact output caps the budget guard priced (cost.py), so the live
    # spend cannot exceed the pre-flight estimate on the dominant (output) axis.
    # --replay opens the cassette read-only and strict: every call must be cached.
    with Cassette(record_path, mode="replay" if args.replay else "record") as cassette:
        complete = cassette.wrap(
            litellm_complete(args.worker_model, max_tokens=worker_max_out), name="worker"
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
                    module_selector=args.module_selector,
                    verify_truth=verify_truth,
                    loop_max_retries=args.loop_max_retries,
                    score_fn=score_fn,
                    feedback_fn=feedback_fn,
                    use_plan_gene=args.plan_gene,
                    stats_fn=text_stats if args.feedback_stats else None,
                    exec_feedback=exec_feedback,
                    critique_complete=critique_complete,
                    oracle_critique=args.oracle_critique,
                    oracle_feedback=oracle_feedback,
                    use_actuator=args.actuator,
                    actuator_context=actuator_context,
                )
            runs.append({name: vars(r) for name, r in results.items()})

    meta = {
        "dataset": dataset,
        "n_train": args.n_train,
        "n_heldout": args.n_heldout,
        "worker_model": args.worker_model,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        # Recorded so a reader can tell whether a multi-gene arm was starved per gene.
        "module_selector": args.module_selector,
        # Recorded so a reader can tell whether the critique gene could see its own
        # behaviour at all (False = the released runs' component-agnostic reflection).
        # Whether the critique gene could see its own behaviour during reflection.
        "detector_reflection": verify_truth is not None,
        "reflection_calls": args.reflection_calls,
        "task_set": args.task_set,
        "loop_max_retries": args.loop_max_retries,
        "plan_gene": args.plan_gene,
        "feedback_stats": args.feedback_stats,
        "critique_model": args.critique_model if not args.oracle_critique else None,
        "oracle_critique": args.oracle_critique,
        "feedback_level": args.feedback_level,
        "exec_feedback": exec_feedback is not None,
        "actuator": args.actuator,
        "worker_max_output_tokens": worker_max_out,
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
