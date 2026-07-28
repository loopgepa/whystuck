"""Render the validation-study report from per-seed run results.

Pure Markdown generation: given a list of per-seed result dicts (each mapping
arm -> serialized ArmResult) plus a `meta` dict, produce `docs/validation-study.md`.
Deterministic — the timestamp comes from `meta["generated"]`, never the clock —
so it's offline-testable and reproducible.

The headline verdict compares arm D (loop-level GEPA) to arm B (prompt-only) on
mean held-out score across seeds, and reports it honestly against seed noise:
SUPPORTED only if D beats B by more than the *summed* per-arm standard deviation
(a deliberately conservative combination, not a statistically pooled estimate).
A single seed has no measurable noise, so a directional verdict needs ≥2 seeds;
with fewer, the verdict is INCONCLUSIVE regardless of the gap.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

ArmResultLike = dict[str, Any]
Run = dict[str, ArmResultLike]

_ARMS = ("A", "B", "C", "D")
_ARM_LABEL = {
    "A": "seed baseline (no optimization)",
    "B": "prompt-only GEPA (one gene)",
    "C": "command control (genome-blind)",
    "D": "loop-level GEPA (whole genome)",
}


def _opt_scores(runs: list[Run], arm: str) -> list[float]:
    return [r[arm]["opt_heldout"] for r in runs if arm in r]


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _verdict(runs: list[Run]) -> tuple[str, str]:
    """Return (label, prose) comparing D vs B on mean held-out across seeds.

    Rule: SUPPORTED iff mean(D) − mean(B) exceeds the summed per-arm stdev;
    INCONCLUSIVE if the gap is positive but within that noise; NOT SUPPORTED if
    D did not beat B. A single seed has zero measured noise (any gap would
    trivially "exceed" it), so <2 seeds on either arm is always INCONCLUSIVE —
    the conservatism only holds once there is variance to measure.
    """
    b, d = _opt_scores(runs, "B"), _opt_scores(runs, "D")
    if not b or not d:
        return "INCONCLUSIVE", "Arms B and D were not both run, so the thesis can't be evaluated."
    if len(b) < 2 or len(d) < 2:
        return (
            "INCONCLUSIVE",
            f"Only {min(len(b), len(d))} seed(s) were run; a single seed has zero measured "
            "noise, so any gap is indistinguishable from chance. Run ≥2 seeds before a "
            "SUPPORTED / NOT SUPPORTED verdict is meaningful.",
        )
    b_mean, d_mean = mean(b), mean(d)
    gap = d_mean - b_mean
    noise = pstdev(b) + pstdev(d)
    if gap > 0 and gap > noise:
        return (
            "SUPPORTED",
            f"Loop-level GEPA (D, mean held-out {_fmt(d_mean)}) beat prompt-only GEPA "
            f"(B, {_fmt(b_mean)}) by Δ={_fmt(gap)}, exceeding the pooled seed noise "
            f"(±{_fmt(noise)}). The thesis holds on this benchmark and budget.",
        )
    if gap > 0:
        return (
            "INCONCLUSIVE",
            f"D ({_fmt(d_mean)}) edged B ({_fmt(b_mean)}) by Δ={_fmt(gap)}, but that gap is "
            f"within the pooled seed noise (±{_fmt(noise)}) — not distinguishable from chance "
            f"at this seed count.",
        )
    return (
        "NOT SUPPORTED",
        f"D ({_fmt(d_mean)}) did not beat B ({_fmt(b_mean)}) (Δ={_fmt(gap)}). On this benchmark "
        f"and budget, loop-level GEPA showed no advantage over prompt-only.",
    )


def _headroom_note(runs: list[Run]) -> str | None:
    """Flag a ceiling/floor that makes the D-vs-B verdict uninformative.

    If the seed baseline already sits near 1.0, no arm can improve, so a mechanical
    NOT SUPPORTED is a saturation artifact, not evidence against the thesis. The
    mirror case (seed near 0) means the tasks may be unsolvable for the worker."""
    baseline = [r[a]["seed_heldout"] for r in runs for a in _ARMS if a in r]
    if not baseline:
        return None
    m = mean(baseline)
    if m >= 0.9:
        return (
            f"⚠ **Ceiling effect.** The seed baseline already scores {_fmt(m)} (near the 1.0 "
            "ceiling), so no arm has **headroom** to improve. This run cannot discriminate "
            "loop-level from prompt-only regardless of the verdict above — read it as "
            "inconclusive by saturation, not as evidence against the thesis. A fair test needs "
            "harder tasks that push the seed well below 1.0."
        )
    if m <= 0.1:
        return (
            f"⚠ **Floor effect.** The seed baseline scores only {_fmt(m)} (near 0), so the tasks "
            "may be unsolvable for this worker — the run has no **headroom** to discriminate the "
            "arms. A fair test needs easier tasks or a more capable worker."
        )
    return None


def _output_cap_note(meta: dict[str, Any]) -> str:
    """Task-set-aware note on whether the worker output cap can truncate answers.

    Code/BCB solutions are long, so the cap is a real (equal-for-every-arm) floor;
    IFBench-style text constraints are short, so the cap never binds. Hardcoding the
    IFBench wording for every task set was a template-drift bug (it misdescribed the
    code and bcb reports, where solutions are exactly why the cap is larger)."""
    if meta.get("task_set") in {"code", "bcb"}:
        return (
            "Code solutions can be long, so this cap can in principle truncate an answer; when it "
            "does, the truncation floor applies to every arm equally, so it cannot bias the "
            "loop-level-vs-prompt-only comparison."
        )
    return (
        "The instruction-following constraints used here have short outputs, so the cap does not "
        "truncate answers; if it ever did, that floor would apply to every arm equally, not bias "
        "the loop."
    )


def _results_table(runs: list[Run]) -> str:
    header = (
        "| Arm | What GEPA evolves | seed held-out | optimized held-out (mean ± std) | Δ |\n"
        "|-----|-------------------|---------------|---------------------------------|---|\n"
    )
    rows = []
    for arm in _ARMS:
        scores = _opt_scores(runs, arm)
        if not scores:
            rows.append(f"| {arm} | {_ARM_LABEL[arm]} | — | Pending | — |")
            continue
        seed_scores = [r[arm]["seed_heldout"] for r in runs if arm in r]
        opt_mean, opt_std = mean(scores), (pstdev(scores) if len(scores) > 1 else 0.0)
        seed_mean = mean(seed_scores)
        delta = opt_mean - seed_mean
        rows.append(
            f"| {arm} | {_ARM_LABEL[arm]} | {_fmt(seed_mean)} | "
            f"{_fmt(opt_mean)} ± {_fmt(opt_std)} | {delta:+.2f} |"
        )
    return header + "\n".join(rows)


def render_report(runs: list[Run], *, meta: dict[str, Any]) -> str:
    """Render the study report as Markdown. Empty `runs` → a 'pending' report."""
    seeds = meta.get("seeds", [])
    method = (
        "## Method\n\n"
        f"- **Dataset:** {meta.get('dataset', 'IFBench')} — "
        f"{meta.get('n_train', '?')} train / {meta.get('n_heldout', '?')} held-out tasks, "
        "scored by programmatic constraint verifiers (no LLM judge).\n"
        f"- **Worker (task LM):** `{meta.get('worker_model', '?')}`; "
        f"**reflection LM:** `{meta.get('reflection_model', '?')}`.\n"
        f"- **Worker loop:** each task eval runs a generate→critique→refine self-refine loop "
        f"(up to {meta.get('loop_max_retries', '?')} refine round(s)). The `generate` gene drives "
        "the draft; the `ensure_correct` gene drives an *intrinsic* critique (the verifier is "
        "never consulted mid-loop — it is the scoring metric, so gating on it would be oracle "
        "contamination). The final answer is scored programmatically. Evolving the verify gene "
        "changes the critique, hence the loop's output — the lever D has and B does not.\n"
        f"- **Budget:** `max_metric_calls`={meta.get('max_metric_calls', '?')} per arm, "
        f"across {len(seeds)} seed(s) {seeds}; estimated spend "
        f"${meta.get('estimated_usd', 0):.2f}.\n"
        "- **Arms:** A = seed baseline (no optimization); B = prompt-only GEPA (evolves one "
        "instruction gene); C = command control (genome-blind worker — the flatness control); "
        "D = loop-level GEPA (evolves the whole genome). All arms share an identical seed "
        "genome, so held-out gaps isolate what GEPA evolves, not prompt length.\n"
    )

    if not runs:
        body = (
            "## Results\n\n**Pending live run (step 7).** The harness, scoring, and budget guard "
            "are built and tested offline; this document fills in once the one paid run records "
            "its cassettes.\n\n" + _results_table(runs) + "\n"
        )
        verdict = "## Verdict\n\nPending the live run.\n"
    else:
        label, prose = _verdict(runs)
        body = "## Results\n\n" + _results_table(runs) + "\n"
        note = _headroom_note(runs)
        verdict = f"## Verdict — {label}\n\n{prose}\n"
        if note:
            verdict += f"\n{note}\n"

    caveats = (
        "## Caveats\n\n"
        "- Arm C's flatness is the genome-blind control; its optimization can't move the score "
        "because the command worker ignores the genome.\n"
        f"- The split is small ({meta.get('n_heldout', '?')} held-out); this claims a *relative* "
        "result (loop-level vs prompt-only) under a matched budget, not a leaderboard number.\n"
        "- Reflection spend is a hard cap: GEPA is told to stop after `reflection_calls` "
        "proposals per arm (one reflection LM call each), so the estimated cost is a genuine "
        "ceiling, not just a guess.\n"
        f"- Worker completions are hard-capped at {meta.get('worker_max_output_tokens', '?')} "
        "tokens (output tokens — the dominant cost — are a genuine ceiling the guard prices "
        "exactly). Input tokens are a conservative estimate sized for GEPA-evolved instructions, "
        "not an absolute bound, so the guard is enforced pre-flight on that estimate rather "
        f"than metered mid-run. {_output_cap_note(meta)}\n"
    )

    return (
        "# WhyStuck validation study: loop-level vs prompt-only GEPA\n\n"
        f"*Generated: {meta.get('generated', 'unknown')}.*\n\n"
        f"{method}\n{body}\n{verdict}\n{caveats}"
    )
