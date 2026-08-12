"""Guards on the measured-spend artifact.

`scripts/measure_actual_spend.py` replays every run with the call surfaces instrumented and prices
the distinct prompts. It takes ~40 minutes, so it is not run in the suite; what IS run is this file,
which reads the committed artifact and checks the three things that make it worth trusting.

The middle one is the reason the whole measurement was worth doing. Every run's metadata carries a
pre-flight `estimated_usd`, and the papers describe it as a **ceiling**. Nothing had ever tested
that it was one. A ceiling nobody checks is a claim, not a guarantee, and it is exactly the shape
of claim these papers are about -- the guard reports "within budget" because it was asked to
compute a bound, never because a bound was compared to an outcome.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs/actual-spend.json"

#: D3's Groq share, measured BY HAND on 2026-08-08 with the critic wrap instrumented, and written
#: into `2026-08-06-true-cross-provider-critic-preregistration.md` before this script existed. The
#: script reproduces it to the token, which is the only external check available on the method:
#: two independent derivations of one number, two days apart, by different means.
D3_HAND_MEASURED = {"calls": 347, "tin": 95_937, "tout": 22_193}
D3_HAND_USD = 0.074


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_method_reproduces_the_independently_hand_measured_d3_share() -> None:
    rows = [r for r in _artifact()["measured"] if r["label"].startswith("D3")]
    assert len(rows) == 1, f"expected exactly one D3 row, found {len(rows)}"
    critic = rows[0]["surfaces"]["critic"]
    for key, expected in D3_HAND_MEASURED.items():
        assert critic[key] == expected, (
            f"D3's critic {key} is {critic[key]}, the hand measurement recorded in the "
            f"pre-registration says {expected}. One of the two is wrong and neither is a typo "
            "worth silently reconciling."
        )
    assert abs(rows[0]["usd_by_provider"]["groq"] - D3_HAND_USD) < 0.0005


def test_no_run_exceeded_its_pre_flight_ceiling() -> None:
    """The bound must actually bound. This is the invariant the artifact exists to test.

    Estimated rows are included: their input side is modelled, so a breach there is weaker
    evidence, but a modelled overrun would still mean the ceiling is too tight to be safe and is
    worth failing on rather than filtering out.
    """
    art = _artifact()
    breaches = [
        (r["label"], r.get("usd", r.get("usd_est")), r["bound_usd"])
        for r in art["measured"] + art["partial"]
        if r["bound_usd"] is not None and r.get("usd", r.get("usd_est", 0.0)) > r["bound_usd"]
    ]
    assert not breaches, (
        "these runs cost more than the pre-flight guard said they could, so the guard is not a "
        f"ceiling: {breaches}. Do not widen the bound to fit -- find out why it was low."
    )


def test_the_totals_are_the_sum_of_the_rows() -> None:
    """A headline that is typed rather than summed is the defect this repository keeps finding."""
    art = _artifact()
    measured = sum(r["usd"] for r in art["measured"])
    estimated = sum(r["usd_est"] for r in art["partial"]) + art["replicates"]["usd_est"]
    assert abs(art["totals"]["measured_usd"] - measured) < 1e-9
    assert abs(art["totals"]["estimated_usd"] - estimated) < 1e-9
    # Measured and estimated must stay separate. Summing them into one field would let a modelled
    # input side be quoted as a measurement, which is the whole reason the two are tracked apart.
    assert "total_usd" not in art["totals"], (
        "a combined total invites quoting a half-modelled figure as measured; keep the split"
    )


def test_the_artifact_is_not_vacuous() -> None:
    art = _artifact()
    assert len(art["measured"]) >= 25, "fewer measured runs than the registry can replay"
    assert len(art["partial"]) >= 18, "the unreplayable runs are missing from the accounting"
    assert art["replicates"]["calls"] > 6000, "the replicate loop's calls are unaccounted for"
    # Every surface ratio used to model the unreplayable inputs must be a real measurement.
    for surface, ratio in art["surface_in_out_ratios"].items():
        assert 0.5 < ratio < 20, f"{surface} in/out ratio of {ratio} is not plausible"
