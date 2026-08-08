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
from collections.abc import Callable, Sequence
from pathlib import Path
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
        "--verdict-model",
        default=None,
        help="verdict/prose splice (E1): route ONLY the PASS/FAIL decision to this model, "
        "leaving the repair text to --critique-model (or the worker). Both see the identical "
        "critique prompt. Exists because a plain model swap moves detection and feedback at "
        "once, so neither half can be credited; this separates them. Adds one LLM call per "
        "critique step. Incompatible with --oracle-critique, whose gate makes no LLM call",
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
        "--skip-preflight",
        action="store_true",
        help="skip the zero-token provider reachability probe that runs before spending. Only "
        "for when the probe itself is wrong (an endpoint moved, a proxy blocks GET but not POST); "
        "it exists because a set-but-unusable key otherwise bills the OTHER provider first",
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
    p.add_argument(
        "--seed-gene",
        action="append",
        default=[],
        metavar="GENE=FILE",
        help="replace one gene's SEED text with the contents of FILE (repeatable, e.g. "
        "`--seed-gene ensure_correct=verify_v2.txt`). The hand-written-gene ablation: it "
        "asks whether a BETTER gene moves held-out at all, which is prior to asking "
        "whether an optimizer can find one. Applies to every arm and does not change "
        "which genes are evolvable, so it composes with A (measure the hand-written gene "
        "alone) or B/D (optimize starting from it). Changes every prompt containing that "
        "gene, so a cassette recorded with one text cannot replay under another",
    )
    p.add_argument(
        "--exclude-constraint",
        action="append",
        default=[],
        metavar="ID",
        help="p0 lever (multi task sets, repeatable): drop a constraint type from the "
        "generated pool. The mechanism this study ends at is a critique that never emits "
        "PASS, which is near-free on the training signal only because most first drafts "
        "fail; raising the base rate of correct first drafts is therefore how that "
        "mechanism gets tested rather than asserted. `exact_word_count` is the useful "
        "knob: it is in every hard task and carries nearly all the residual. Changes "
        "every task, hence every cassette key, so a run recorded with an exclusion "
        "cannot replay without it",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="pin the worker sampling temperature (default: the provider's, which is "
        "what every committed run used). Required before repeated-sample intervals mean "
        "anything: no interval in this study contains within-condition sampling "
        "variance, because the worker draws one sample per task at an unpinned "
        "temperature. Passed per call so it enters the cassette key -- a factory-level "
        "temperature would let two runs at different temperatures serve each other's "
        "completions",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="replicate index, used ONLY as a cassette-key salt (never sent to the provider). "
        "Repeated scorings of one genome are identical calls, so record mode dedupes them and "
        "K 'independent' samples come back byte-identical -- reporting ZERO sampling variance, "
        "the most flattering artifact available to this study. Pass a distinct index per "
        "replicate. Deliberately independent of --temperature: a faithful estimate of the "
        "noise in the ALREADY-REPORTED numbers must draw from the same distribution they were "
        "drawn from, which is the provider's default, so pin nothing",
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
        default=None,
        help="where to write the Markdown report (default: none, symmetric with --out). "
        "This defaulted to docs/validation-study.md, which meant any run or replay that "
        "omitted the flag silently overwrote a COMMITTED report belonging to a different "
        "run; nothing depended on the default (verify_reproducibility.py and REPRODUCE.md "
        "both pass it explicitly), so the only thing it did was clobber",
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


def _parse_gene_overrides(specs: list[str], *, active: Sequence[str]) -> dict[str, str]:
    """Parse `--seed-gene GENE=FILE` specs into {gene_id: text}.

    Every failure mode raises rather than degrading, because each one silently runs a
    DIFFERENT experiment than the operator asked for and bills for it: a gene id that is
    not live in this configuration (e.g. `actuator` without `--actuator`) would run the
    un-ablated condition; a blank file would send the model a prompt with no instruction,
    which scores like a bad instruction rather than announcing itself; and the same gene
    given twice would let last-wins pick one of two requested conditions at random.

    Kept a pure function of (specs, active) so the parsing and refusal rules are testable
    without a run — the flag exists to make a paid measurement trustworthy, so its own
    guards must not be reachable only through a paid path.
    """
    overrides: dict[str, str] = {}
    for spec in specs:
        gene, sep, raw_path = spec.partition("=")
        if not sep or not gene.strip() or not raw_path.strip():
            raise ValueError(f"--seed-gene {spec!r} is not of the form GENE=FILE")
        gene = gene.strip()
        if gene not in active:
            raise ValueError(
                f"--seed-gene {gene!r} names no gene in this run's genome "
                f"(live genes: {', '.join(active)}); enable the lever that adds it "
                "(--plan-gene / --actuator) or fix the id"
            )
        if gene in overrides:
            raise ValueError(f"--seed-gene {gene!r} given twice; pick one file per gene")
        path = Path(raw_path.strip())
        if not path.is_file():
            raise ValueError(f"--seed-gene {gene}: no such file {str(path)!r}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(
                f"--seed-gene {gene}: {str(path)!r} is blank; a blank gene does not disable "
                "its step, it sends the model a prompt with no instruction"
            )
        overrides[gene] = text
    return overrides


#: litellm reads each vendor's credential from its own environment variable. Kept here so the
#: pre-spend check below covers whatever provider a run actually names, rather than assuming
#: OpenAI. A provider absent from this map is not gated -- add it when you add its price.
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

#: A zero-token endpoint per provider that answers 200 iff the credential works AND the host can
#: reach the API. (url_template, header_builder). Listing models bills nothing anywhere.
_PROVIDER_PROBE: dict[str, tuple[str, Callable[[str], dict[str, str]]]] = {
    "openai": ("https://api.openai.com/v1/models", lambda k: {"Authorization": f"Bearer {k}"}),
    "groq": ("https://api.groq.com/openai/v1/models", lambda k: {"Authorization": f"Bearer {k}"}),
    "anthropic": (
        "https://api.anthropic.com/v1/models",
        lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"},
    ),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models?key={key}", lambda k: {}),
}


def probe_provider(provider: str, key: str, *, timeout: float = 15.0) -> tuple[bool, str]:
    """(reachable-and-authorized, human-readable status) for one provider, at zero token cost.

    WHY THIS IS NOT JUST A KEY-PRESENCE CHECK. A set-but-unusable credential is worse than a
    missing one: the run passes the cheap gate, spends the WORKER's budget, and dies on the first
    call to the other provider, leaving a bill and a partial cassette. That is not hypothetical --
    on 2026-08-06 the D3 critic's provider returned 403 at its edge for this host, and only the
    absence of the key stopped a live attempt.

    It also separates failures the caller must act on differently, which a mid-run exception does
    not: 401/403 is a credential or network refusal, 429 is a key that works and has no quota left.
    """
    import urllib.error
    import urllib.request

    if provider not in _PROVIDER_PROBE:
        return True, "no probe defined; not checked"
    template, headers = _PROVIDER_PROBE[provider]
    # An explicit User-Agent, because the default one is a liability. `urllib` sends
    # `Python-urllib/3.x`, and on 2026-08-06 Groq's edge returned 403 to exactly that string while
    # returning 200 to curl and to litellm from the same host, seconds apart. A preflight that
    # false-negatives is worse than none: it blocks runs that would have worked, and it teaches the
    # operator to reach for --skip-preflight, which disables the check that catches the real thing.
    request = urllib.request.Request(  # noqa: S310 - fixed https literals above
        template.format(key=key),
        headers={"User-Agent": "whystuck-validation-study/1.0", **headers(key)},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        hint = {
            401: "credential rejected",
            403: "refused before auth (network/region block) or credential rejected",
            429: "credential works, quota or rate limit exhausted",
        }.get(exc.code, "")
        return False, f"HTTP {exc.code}{f' - {hint}' if hint else ''}"
    except Exception as exc:  # noqa: BLE001 - any transport failure is the same verdict here
        return False, f"unreachable: {type(exc).__name__}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    arms = tuple(args.arms)

    # The oracle gate makes no critique LLM call, so there is no verdict to route.
    # Accepting the pair would ignore one flag and record metadata claiming a splice
    # that never ran -- this paper series' own failure mode, as a CLI slip.
    if args.verdict_model and args.oracle_critique:
        print(
            "ERROR: --verdict-model cannot be combined with --oracle-critique: the oracle "
            "gate decides by the true metric and issues no critique LLM call, so no verdict "
            "would be routed and the recorded run would not be the spliced one.",
            file=sys.stderr,
        )
        return 2

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
    if args.verdict_model:
        model_args.append(("--verdict-model", args.verdict_model))
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

    # p0-lever guard. An id that names no live constraint would leave the pool untouched and
    # run the UN-ablated condition under an ablated label -- the same failure as a gene
    # override naming a gene that is not live, so it is refused the same way.
    if args.exclude_constraint:
        if args.task_set not in ("multi", "multi-hard"):
            print(
                f"ERROR: --exclude-constraint needs --task-set multi or multi-hard; "
                f"{args.task_set!r} draws its tasks from a vendored file, not a generator.",
                file=sys.stderr,
            )
            return 2
        from whystuck.eval.multi_constraint import _EASY_IDS, _HARD_IDS

        live = set(_HARD_IDS if args.task_set == "multi-hard" else _EASY_IDS)
        unknown = sorted(set(args.exclude_constraint) - live)
        if unknown:
            print(
                f"ERROR: --exclude-constraint {unknown} names no constraint in the "
                f"{args.task_set!r} pool (live: {', '.join(sorted(live))}). Excluding an "
                "unknown id would silently run the un-ablated condition under an ablated "
                "label.",
                file=sys.stderr,
            )
            return 2

    # Starved-budget guard. GEPA charges its seed evaluation len(valset) metric calls
    # (gepa/core/state.py sets total_num_evals = len(seed_valset_evaluation.scores_by_val_id)),
    # and optimize_loop passes the train split as the trainset with no separate valset, so
    # valset size IS n_train. One accept cycle therefore needs the seed evaluation, a
    # reflection minibatch (GEPA's default 3, capped at the split size), and a second full
    # evaluation before a proposal can enter the Pareto front.
    #
    # Below that, the arm still runs, still bills, and still reports -- as genes_changed=[]
    # and a flat delta. That is indistinguishable from "optimization did not help" while
    # being "optimization never happened", which is the precise defect this study had to
    # retract a result over. So refuse at the dry-run step instead of pricing it.
    #
    # Scoped to arms that actually optimize: the released C1 conditions run `--arms A` at
    # n_train=32 with mmc=64, where GEPA never runs and the budget is inert.
    from whystuck.eval.study import arm_evolvable

    optimizing = [
        a for a in arms if arm_evolvable(a, include_plan=args.plan_gene, actuator=args.actuator)
    ]
    min_calls = 2 * args.n_train + min(3, args.n_train)
    if optimizing and args.max_metric_calls < min_calls:
        print(
            f"ERROR: --max-metric-calls {args.max_metric_calls} cannot fund one accept "
            f"cycle at --n-train {args.n_train}. GEPA spends {args.n_train} metric calls on "
            f"the seed evaluation alone (its valset is the train split), then needs a "
            f"minibatch of {min(3, args.n_train)} plus a further {args.n_train} to promote a "
            f"proposal: {min_calls} minimum for arms {''.join(optimizing)}. Below it the arm "
            f"reports genes_changed=[] and a flat delta, which reads as 'optimization did not "
            f"help' when it means 'optimization never ran'. Raise --max-metric-calls to "
            f">= {min_calls} (holding metric calls per train task constant across a "
            f"split-size sweep means scaling it WITH --n-train), or lower --n-train.",
            file=sys.stderr,
        )
        return 2

    # Hand-written gene overrides. Validated here, at the cheap dry-run step, and against
    # the genes THIS configuration actually builds — derived from `seed_genome` itself so
    # the live-gene set cannot drift from what the study constructs.
    from whystuck.eval.study import seed_genome

    live_genes = [
        g.id
        for g in seed_genome(
            "probe",
            evolvable=set(),
            include_plan=args.plan_gene,
            include_actuator=args.actuator,
        ).genes
    ]
    try:
        gene_overrides = _parse_gene_overrides(args.seed_gene, active=live_genes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for gene, text in gene_overrides.items():
        print(f"Seed gene override [{gene}]: {text.strip()!r}")

    # Call count per eval: generate (+plan) + per retry a refine, plus a critique
    # UNLESS the oracle gate replaces it (no critique LLM call then), plus an actuator
    # call per retry when the actuator gene is on (critique + actuator + refine = 3).
    if args.oracle_critique:
        per_retry_calls = 1
    elif args.actuator:
        per_retry_calls = 3
    elif args.verdict_model:
        # Splice (E1): the verdict and the prose are two calls on the same prompt,
        # so a retry costs verdict + prose + refine.
        per_retry_calls = 3
    else:
        per_retry_calls = 2
    worker_calls_per_task = (2 if args.plan_gene else 1) + per_retry_calls * args.loop_max_retries
    # Conservative pricing: a cross-family critic, and under the splice a separate
    # verdict model, run some worker-side calls on a different (often pricier) model.
    # We price ALL worker-side calls at the priciest model any of them can route to, so
    # the guard's estimate stays an upper bound (the true split is cheaper). Irrelevant
    # under --oracle-critique (no critique call, hence no verdict call either).
    #
    # The verdict model belongs in this max for the same reason the critic does, and it
    # was missed once: a cheap-prose / expensive-verdict cell then priced every call at
    # the cheap worker and the ceiling silently sat below the true cost.
    pricing_worker_model = args.worker_model
    if not args.oracle_critique:
        # Pick the pricier model by TRUE cost at the real worker call shape (input-heavy),
        # not by unweighted sum(in,out) — a future entry with an unusual in/out ratio
        # could flip a sum-based comparison and silently under-price the guard.
        cost_at_shape = lambda m: CallGroup(m, 1, *WORKER_TOKENS).usd()  # noqa: E731
        routed = [m for m in (args.critique_model, args.verdict_model) if m]
        pricing_worker_model = max([args.worker_model, *routed], key=cost_at_shape)

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
        return _run_live(args, arms, seeds, estimate, gene_overrides)

    if not args.yes:
        print("\nDry run only. Re-run with --yes (and OPENAI_API_KEY set) to spend.")
        return 0

    # Every provider this run will actually call, not just OpenAI. A cross-provider critic
    # (D3) routes one call per eval to a second vendor, and checking only OPENAI_API_KEY
    # meant that run would spend the worker's money and THEN die on the missing second key,
    # mid-study, with a partial cassette. Same rule as every other validation in this file:
    # refuse at the cheap step. The check is on presence, not validity -- a live probe would
    # cost a call and could not be done offline -- so it catches the unset key, not the
    # expired one.
    models = [m for m in (args.worker_model, args.reflection_model, args.critique_model) if m]
    providers = sorted({m.split("/", 1)[0] for m in models})
    missing = sorted(
        {
            var
            for provider in providers
            for var in [_PROVIDER_KEY_ENV.get(provider)]
            if var and not os.environ.get(var)
        }
    )
    if missing:
        print(
            f"ERROR: --yes given but these provider keys are not set: {', '.join(missing)}. "
            f"This run calls {', '.join(providers)}.",
            file=sys.stderr,
        )
        return 3

    if not args.skip_preflight:
        unusable = []
        for provider in providers:
            var = _PROVIDER_KEY_ENV.get(provider)
            ok, status = probe_provider(provider, os.environ.get(var or "", ""))
            print(f"  preflight {provider:<10} {status}")
            if not ok:
                unusable.append(f"{provider} ({status})")
        if unusable:
            print(
                f"ERROR: --yes given but these providers did not answer: {', '.join(unusable)}. "
                "Nothing was spent. A run that starts here bills the reachable provider and then "
                "dies on the first call to this one, leaving a partial cassette. "
                "Override with --skip-preflight only if you know the probe is wrong.",
                file=sys.stderr,
            )
            return 4

    return _run_live(args, arms, seeds, estimate, gene_overrides)


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
    args: argparse.Namespace,
    arms: tuple[str, ...],
    seeds: list[int],
    estimate: float,
    gene_overrides: dict[str, str],
) -> int:
    """The live run (reached only with --yes + a key). Not exercised offline."""
    import tempfile

    from whystuck.core.storage import WhyStuckStore
    from whystuck.core.util import utc_now_iso
    from whystuck.eval.bcb_bench import bcb_feedback, bcb_score, load_bcb_tasks
    from whystuck.eval.benchmark import seeds_are_independent
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
        mc_exclude = frozenset(args.exclude_constraint)

        def load_split(split: str, seed: int, n: int):
            return load_multi_constraint(  # type: ignore[arg-type]
                split, seed=seed, n=n, pool=mc_pool, hard=hard, exclude=mc_exclude
            )

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
        # Add the kwargs at the call site, OUTSIDE the cassette wrapper, so they land in
        # the hashed kwargs. Absent both flags the call is unchanged and every existing
        # cassette key still resolves. `sample` is a salt: it never reaches the provider,
        # it only stops record mode from serving one cached draw to K replicates.
        _extra: dict[str, object] = {}
        if args.temperature is not None:
            _extra["temperature"] = args.temperature
        if args.sample is not None:
            _extra["sample"] = args.sample

        def _sampled(fn: Callable[..., str]) -> Callable[[str], str]:
            """Apply the run's sampling kwargs to one wrapped model callable.

            Every sampled surface goes through this. It used to be inlined on the worker
            alone, so `--temperature 0` pinned the worker and left the critique and verdict
            calls at the provider default while the run metadata recorded a temperature --
            a flag reported as applied that was not. It surfaced only because two calls on
            one prompt at temperature 0 came back different 30 times out of 30.
            """
            if not _extra:
                return fn
            return lambda prompt: fn(prompt, **_extra)

        complete = _sampled(complete)

        reflection = cassette.wrap(
            litellm_complete(args.reflection_model, max_tokens=REFLECTION_MAX_OUTPUT_TOKENS),
            name="reflection",
        )
        # Cross-family critic (run 6): a separate model for the critique step only.
        # Skipped under --oracle-critique (the gate makes no critique LLM call).
        critique_complete = None
        if args.critique_model and not args.oracle_critique:
            critique_complete = _sampled(
                cassette.wrap(
                    litellm_complete(args.critique_model, max_tokens=WORKER_MAX_OUTPUT_TOKENS),
                    name="critic",
                )
            )
        # Verdict/prose splice (E1). Its own cassette namespace: the verdict call shares the
        # critique call's prompt exactly, so a shared namespace would collide the two whenever
        # both route to the same model -- returning the prose model's text as the verdict and
        # silently un-splicing the experiment.
        verdict_complete = None
        if args.verdict_model and not args.oracle_critique:
            verdict_complete = _sampled(
                cassette.wrap(
                    litellm_complete(args.verdict_model, max_tokens=WORKER_MAX_OUTPUT_TOKENS),
                    name="verdict",
                )
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
                    verdict_complete=verdict_complete,
                    oracle_critique=args.oracle_critique,
                    oracle_feedback=oracle_feedback,
                    use_actuator=args.actuator,
                    actuator_context=actuator_context,
                    gene_overrides=gene_overrides,
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
        # Whether this task set regenerates its pool per seed. False means a seed only
        # picks a partition of one fixed pool, so cross-seed intervals and sign tests are
        # anti-conservative — the analysis must know without re-deriving it from the loader.
        "seeds_independent": seeds_are_independent(args.task_set),
        # Recorded so a reader can tell whether the critique gene could see its own
        # behaviour at all (False = the released runs' component-agnostic reflection).
        # Whether the critique gene could see its own behaviour during reflection.
        "detector_reflection": verify_truth is not None,
        "reflection_calls": args.reflection_calls,
        "task_set": args.task_set,
        "temperature": args.temperature,
        # The replicate salt. Recorded because two runs differing only in their salt are
        # otherwise indistinguishable in the release, and a repeated-sample interval whose
        # replicates cannot be told apart is not checkable.
        "sample": args.sample,
        "exclude_constraint": sorted(args.exclude_constraint),
        "loop_max_retries": args.loop_max_retries,
        "plan_gene": args.plan_gene,
        "feedback_stats": args.feedback_stats,
        "critique_model": args.critique_model if not args.oracle_critique else None,
        "verdict_model": args.verdict_model if not args.oracle_critique else None,
        "oracle_critique": args.oracle_critique,
        "feedback_level": args.feedback_level,
        "exec_feedback": exec_feedback is not None,
        "actuator": args.actuator,
        # The hand-written gene text in full, not a flag: it IS the experimental
        # condition, so a released result must carry the exact prompt that produced it.
        "gene_overrides": gene_overrides,
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
