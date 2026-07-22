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

    # the real run — default is Terra × 2 seeds ≈ $3.43 (needs ≥2 seeds for a
    # verdict; a single seed is always INCONCLUSIVE). Sol × 2 seeds ≈ $5.38
    # exceeds the default $5 cap — raise --max-usd or lower --max-metric-calls.
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
    BudgetExceeded,
    CostGuard,
    estimate_usd,
    plan_study_calls,
)

_VALID_ARMS = frozenset("ABCD")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run/dry-run the GEPA validation study.")
    p.add_argument("--max-usd", type=float, default=5.0, help="hard spend ceiling (default 5.0)")
    p.add_argument("--arms", default="ABCD", help="which arms to run (default ABCD)")
    p.add_argument("--n-train", type=int, default=31, help="train task count (default 31)")
    p.add_argument("--n-heldout", type=int, default=31, help="held-out task count (default 31)")
    p.add_argument("--max-metric-calls", type=int, default=150, help="GEPA budget per arm")
    p.add_argument("--reflection-calls", type=int, default=15, help="est. reflection calls/arm")
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
    for label, model in (
        ("--worker-model", args.worker_model),
        ("--reflection-model", args.reflection_model),
    ):
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

    groups = plan_study_calls(
        arms=arms,
        n_train=args.n_train,
        n_heldout=args.n_heldout,
        max_metric_calls=args.max_metric_calls,
        reflection_calls=args.reflection_calls,
        worker_model=args.worker_model,
        reflection_model=args.reflection_model,
        n_seeds=len(seeds),
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
    from whystuck.eval.ifbench import load_ifbench
    from whystuck.eval.llm_worker import litellm_complete
    from whystuck.eval.report import render_report
    from whystuck.eval.study import run_study

    record_path = args.record or "study-cassette.json"
    runs: list[dict[str, dict]] = []
    with Cassette(record_path, mode="record") as cassette:
        complete = cassette.wrap(litellm_complete(args.worker_model), name="worker")
        reflection = cassette.wrap(litellm_complete(args.reflection_model), name="reflection")
        for seed in seeds:
            train = load_ifbench(split="train", seed=seed, n=args.n_train)
            heldout = load_ifbench(split="heldout", seed=seed, n=args.n_heldout)
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
                )
            runs.append({name: vars(r) for name, r in results.items()})

    meta = {
        "dataset": "IFBench",
        "n_train": args.n_train,
        "n_heldout": args.n_heldout,
        "worker_model": args.worker_model,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
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
