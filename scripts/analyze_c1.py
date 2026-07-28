#!/usr/bin/env python
"""Task-level paired analysis of the C1 verifier ablation.

**Pre-registered: written and committed before the run was paid for**, so the test could
not be chosen after seeing the numbers. The two conditions differ only in the
`ensure_correct` gene (seed rubric vs `docs/genes/verify_measure_v2.txt`); everything else
-- task set, seed, split, budget, worker -- is identical, and because the cassette is
content-addressed and shared, both conditions reuse the *same first draft* for each task.
So each pair isolates the verify gene: same draft in, different critique, different repair.

The unit of analysis is the **task**, not the seed. Seeds on the vendored families merely
re-partition one fixed pool (see `benchmark.seeds_are_independent`), and even on the
generated families a 3-seed sign test cannot fall below p = 0.25. Thirty-two paired tasks
can.

Two statistics, reported together on purpose:

  * an exact two-sided **sign test** on the non-zero paired differences. Answers "did the
    score move at all". This is the decision-relevant question for C1, in EITHER direction:
    if rewriting the verify gene's prose can move held-out, an optimizer restricted to
    rewriting prose has a gradient to climb, and the optimization run is worth buying. A
    negative effect would mean the gene matters and this particular rubric was bad, which
    is still signal.
  * a deterministic percentile **bootstrap** interval on the mean difference. Exact
    enumeration is what the seed-level analysis uses, but at n = 32 that is 32**32
    resamples, so this is Monte Carlo with a fixed seed and a stated B.

Reported as a 2x2, so that "not significant" can never be read as "no effect":

                      | equivalent (90% CI within +-SESOI) | not equivalent
  significant         | MOVES, TRIVIALLY                   | MOVES
  not significant     | NO EFFECT                          | INCONCLUSIVE (underpowered)

    python scripts/analyze_c1.py docs/validation-study-c1-base.json \\
                                 docs/validation-study-c1-hand.json
"""

from __future__ import annotations

import json
import random
import sys
from statistics import mean

from analyze_deltas import sign_test_p

#: Smallest effect size of interest, on the mean fraction-of-constraints-satisfied scale.
#: Anchored to the study's own measured effects: the actionability lever moved the baseline
#: +0.063 and the tool-fed critique +0.072, so an effect below 0.05 is smaller than the
#: levers the paper already calls real, and calling it a lift would be over-reading.
SESOI = 0.05

#: Bootstrap resamples, and the RNG seed. Fixed so the interval is reproducible to the
#: digit; the seed is 0 rather than chosen, and B is large enough that the 2.5% percentile
#: is stable to ~0.001 at this n.
BOOTSTRAP_B = 100_000
BOOTSTRAP_SEED = 0


def _arm(path: str, arm: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    runs = payload["runs"]
    if len(runs) != 1:
        raise ValueError(
            f"{path} holds {len(runs)} seeds; C1 is a single-seed, task-paired design, and "
            "collapsing seeds here would silently reintroduce the seed as the unit"
        )
    return runs[0][arm]


def paired_by_task(base_json: str, hand_json: str, arm: str = "A") -> dict[str, float]:
    """{task_id: hand - base} on the per-task held-out scores.

    Raises unless the two conditions scored exactly the same task set: the whole design
    rests on the pairing, and a silent intersection would quietly change the sample.
    Arm A measures held-out twice (its determinism check), and under a content-addressed
    cassette the second pass must agree with the first; we verify that rather than assume
    it, because a disagreement would mean the run was not deterministic and no paired test
    on it is meaningful.
    """
    out: dict[str, float] = {}
    base, hand = _arm(base_json, arm), _arm(hand_json, arm)
    for label, res in (("base", base), ("hand", hand)):
        first, second = res["seed_heldout_by_task"], res["opt_heldout_by_task"]
        if not first:
            raise ValueError(
                f"{label} condition carries no per-task scores; it predates "
                "ArmResult.seed_heldout_by_task and cannot be analysed task-wise"
            )
        if first != second:
            differing = sorted(k for k in first if first.get(k) != second.get(k))
            raise ValueError(
                f"{label} condition is not deterministic: {len(differing)} task(s) scored "
                f"differently on the second held-out pass (e.g. {differing[:3]})"
            )
    if set(base["seed_heldout_by_task"]) != set(hand["seed_heldout_by_task"]):
        raise ValueError(
            "the two conditions scored different task sets, so they cannot be paired; "
            "check that task-set, seed and n-heldout match exactly"
        )
    for task_id, base_score in base["seed_heldout_by_task"].items():
        out[task_id] = hand["seed_heldout_by_task"][task_id] - base_score
    return out


def bootstrap_ci(
    diffs: list[float], alpha: float = 0.05, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap on the mean paired difference.

    Monte Carlo rather than exact because n = 32 gives 32**32 resamples. Deterministic
    given (diffs, b, seed). Percentile index is `ceil(q*b) - 1`, matching the exact
    estimator's convention so the two are read the same way.
    """
    if not diffs:
        raise ValueError("no differences given")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean(rng.choices(diffs, k=n)) for _ in range(b))
    import math

    lo = means[math.ceil((alpha / 2) * b) - 1]
    hi = means[math.ceil((1 - alpha / 2) * b) - 1]
    return mean(diffs), lo, hi


def verdict(diffs: list[float], sesoi: float = SESOI) -> tuple[str, dict]:
    """The pre-registered 2x2 call, plus the statistics behind it.

    Equivalence uses the two-sided 90% interval, which is the interval-based form of a
    TOST at alpha = 0.05: if it lies entirely inside +-sesoi, both one-sided tests reject.
    Keeping significance and equivalence as separate axes is the point -- a wide interval
    straddling zero is *underpowered*, not evidence of no effect, and this study has
    already published one claim that confused the two.
    """
    p = sign_test_p(diffs)
    m, lo95, hi95 = bootstrap_ci(diffs, alpha=0.05)
    _, lo90, hi90 = bootstrap_ci(diffs, alpha=0.10)
    significant = p < 0.05
    equivalent = -sesoi < lo90 and hi90 < sesoi
    label = {
        (True, True): f"MOVES, BUT TRIVIALLY (|effect| < {sesoi})",
        (True, False): "MOVES",
        (False, True): f"NO EFFECT (equivalent within +-{sesoi})",
        (False, False): "INCONCLUSIVE (underpowered: interval straddles zero and SESOI)",
    }[(significant, equivalent)]
    stats = {
        "n": len(diffs),
        "ties": sum(1 for d in diffs if d == 0.0),
        "mean": m,
        "ci95": (lo95, hi95),
        "ci90": (lo90, hi90),
        "sign_p": p,
        "significant": significant,
        "equivalent": equivalent,
        "wins_hand": sum(1 for d in diffs if d > 0),
        "wins_base": sum(1 for d in diffs if d < 0),
    }
    return label, stats


def main(argv: list[str]) -> int:  # pragma: no cover - CLI convenience
    if len(argv) != 2:
        print(__doc__)
        return 2
    diffs_by_task = paired_by_task(argv[0], argv[1])
    diffs = list(diffs_by_task.values())
    label, s = verdict(diffs)
    print("=== C1 verifier ablation: hand-written rubric - seed rubric, paired by TASK ===")
    print(
        f"n={s['n']} tasks   ties={s['ties']}   hand better on {s['wins_hand']}, "
        f"seed better on {s['wins_base']}"
    )
    print(
        f"mean Δ={s['mean']:+.4f}   95% CI [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]   "
        f"90% CI [{s['ci90'][0]:+.4f},{s['ci90'][1]:+.4f}]"
    )
    print(
        f"exact two-sided sign test p={s['sign_p']:.4f}   (SESOI ±{SESOI}, "
        f"bootstrap B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED})"
    )
    print(f"\nVERDICT: {label}")
    print(
        "\nReading: C1 asks whether rewriting this gene's PROSE can move held-out at all,\n"
        "in either direction. MOVES => an optimizer restricted to prose has a gradient, so\n"
        "the optimization run is worth buying. NO EFFECT => prose-only detector improvement\n"
        "is inert here and a loop-level null says nothing about the optimizer."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
