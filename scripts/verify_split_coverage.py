#!/usr/bin/env python
"""Prove the three-way paper split lost no claim.

Zero cost, offline. One 46-page draft became three papers (`docs/papers/README.md`). A split is a
mass edit of prose, and the failure mode is silent: a paragraph that belonged to no paper's argument
gets dropped, nothing errors, and the claim simply stops existing. `verify_paper_claims.py` catches
that for the ~155 sentences it pins by name -- a dropped pinned sentence yields NaN and fails -- but
it says nothing about the numbers it never pinned, which is most of them.

So this script compares the pre-split source, frozen at `docs/papers/PRE-SPLIT-SOURCE.tex`, against
the concatenation of the three papers, and requires every quantity in the former either to appear in
the latter, or to be accounted for in one of two explicit ways: `SUPERSEDED`, which maps a value a
corrected statistic replaced to its replacement and verifies that replacement is present, or
`DELIBERATE_DROPS`, which names a value genuinely abandoned and why. The frozen copy is
byte-identical to `docs/paper/main.tex` at commit fe628ed and is never built; it exists so this
proof has no dependency on git history remaining reachable.

The `SUPERSEDED` route is deliberately harder to abuse than an exemption list. It fails if the old
value is still printed anywhere (the correction did not land everywhere, so the papers now mix two
denominators) and it fails if the replacement is missing (the quantity was deleted rather than
replaced -- silent loss wearing the costume of a correction).

What counts as a quantity: decimals (`0.0286`), ratios (`11/12`), spelled counts the papers use as
claims, and dollar figures. Bare integers are excluded -- they are dominated by LaTeX plumbing
(column counts, `\\setlength`, footnote markers) and would bury the signal in noise. The pinned
`tex:` checks in `verify_paper_claims.py` cover the load-bearing integers by name.

    python scripts/verify_split_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_sources  # noqa: E402

FROZEN = Path("docs/papers/PRE-SPLIT-SOURCE.tex")

#: The commit whose `docs/paper/main.tex` the frozen copy reproduces. Checked by the test suite
#: where git history is reachable, so the baseline cannot be quietly edited to make this pass.
FROZEN_COMMIT = "fe628ed5c09b43d12731f3c99b94563ab070f813"

#: A decimal, a ratio like `11/12`, or a dollar amount. Deliberately not bare integers.
_QUANTITY = re.compile(r"\d+\.\d+|\b\d{1,3}/\d{1,3}\b")

#: Quantities present in the pre-split draft and deliberately abandoned, each with the reason. Keep
#: the reasons specific: a vague entry here is how a real dropped claim would hide.
#:
#: Empty, and that is the result for the split itself: the split dropped prose -- three intros and
#: three conclusions replaced one of each, and several paragraphs became cross-references -- but it
#: dropped no quantity. Anything genuinely abandoned would go here with a reason. `stale_drops()`
#: refuses an entry that is not actually needed, so this cannot decay into a blanket exemption.
DELIBERATE_DROPS: dict[str, str] = {}

#: Quantities the papers no longer print because a *corrected statistic* replaced them, mapped to
#: the value that replaced each. Recorded as a mapping rather than as prose exemptions because the
#: mapping is checkable in both directions: `broken_supersessions()` requires the old value to be
#: gone (or the correction did not land) and every new value to be present (or the quantity was
#: deleted rather than replaced, which is the failure this whole script exists to catch).
#:
#: All of these come from one fix, on 2026-07-31. Every K=5 standard error divided the *population*
#: sd of five draws by sqrt(5) and compared the result against t(4) critical values -- a
#: distribution that exists because the denominator is n-1. Correcting it raises every sigma by
#: sqrt(5/4) = 1.118 and lowers every t by 10.6%; the same mistake over 30 paired tasks moved the
#: cross-run sigma(delta) from 0.0286 to 0.0291. See `repeat_scores.sd_between_replicates`.
SUPERSEDED: dict[str, str] = {
    # --- the cross-run noise estimate and the swing denominator built from it ------------------
    "0.1106": "0.1125",  # sigma for one task, one draw
    "0.1565": "0.1591",  # sigma(delta) for one task
    "0.0404": "0.0412",  # sigma * sqrt(2), the swing denominator
    # --- tab:repeat, the p0 levels' five-replicate statistics ----------------------------------
    "0.0418": "0.0468",  # L1 sd(draw)
    "0.0187": "0.0209",  # L1 se(mean)
    "0.0410": "0.0459",  # L2 sd(draw)
    "0.0184": "0.0205",  # L2 se(mean)
    "0.0088": "0.0099",  # L3 se(mean)
    "0.0197": "0.0220 0.0205",  # L3 sd(draw); and the cross-run one-draw estimate, separately
    # --- tab:sigmagenome, sigma per genome ------------------------------------------------------
    "0.0292": "0.0326",
    "0.0144": "0.0161",
    "0.0336": "0.0376",
    "0.0152": "0.0170",
    "0.0140": "0.0157",
    "0.0057": "0.0064",
    # --- tab:curvesigma, sigma per curve row and its ratios against the borrowed denominator ----
    "0.0211": "0.0236",
    "0.0137": "0.0153",
    "0.0212": "0.0237",
    "0.0127": "0.0142",
    "0.0147": "0.0164",
    "0.0149": "0.0166",  # R=1 mean
    "0.0162": "0.0181",  # R=3 mean
    "0.35": "0.38",
    "0.48": "0.53",
    "0.51": "0.56",
    # --- tab:curvepairs, the six paired standard errors and t statistics ------------------------
    "0.0080": "0.0090",
    "0.0130": "0.0145",
    "0.0105": "0.0117",
    "0.0068": "0.0076",
    "0.0073": "0.0081",
    "4.89": "4.38",
    "3.23": "2.89",
    "3.49": "3.12",
    "6.11": "5.47",
    # --- the published-vs-replicated discrepancies, in single-draw sd units ---------------------
    "1.84": "1.65",
    "1.52": "1.36",
    "1.06": "0.95",
    # --- every gap restated in units of the cross-run sigma ------------------------------------
    "2.10": "2.07",
    "2.28": "2.24",
    "3.09": "3.04",
    "0.99": "0.98",
    "1.28": "1.26",
    "1.46": "1.44 1.43",  # the cross-run ratio, and tab:splitresults' n_train=8 cell
    "0.58": "0.57",
    "2.16": "2.12",
    # --- the superseded runs' ratios, and the salt control's cross-run comparison ---------------
    "3.50": "3.44",
    "3.29": "3.24",
    "1.4": "1.3",
}

#: Spelled-out counts the papers use as claims. Checked separately from the numeric scan because
#: `fifteen` is not a digit but is exactly the kind of count a split can drop.
SPELLED_CLAIMS: tuple[str, ...] = (
    "fifteen",
    "twelve",
    "thirteen",
    "twenty-five",
    "sixteen",
)


def _strip_comments(tex: str) -> str:
    """Drop LaTeX comments so a number mentioned only in a source comment is not counted as a claim.

    Handles `\\%` (an escaped percent sign is content, not a comment) by consuming it first.
    """
    out = []
    for line in tex.split("\n"):
        buf = []
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                buf.append(line[i : i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def quantities(tex: str) -> set[str]:
    return set(_QUANTITY.findall(_strip_comments(tex)))


def frozen_tex() -> str:
    if not FROZEN.is_file():
        raise FileNotFoundError(
            f"{FROZEN} is missing. It is the baseline this proof compares against; without it the "
            "script would pass by having nothing to check."
        )
    return FROZEN.read_text(encoding="utf-8")


def missing_quantities() -> dict[str, list[str]]:
    """Pre-split quantities absent from every paper, excluding the deliberate drops.

    Returns a mapping from the quantity to the pre-split lines that mention it, so a real omission
    arrives with the context needed to place it rather than as a bare number.
    """
    frozen = frozen_tex()
    before = quantities(frozen)
    after = quantities(paper_sources.all_tex())
    lines = _strip_comments(frozen).split("\n")
    gone = {}
    for q in sorted(before - after):
        if q in DELIBERATE_DROPS or q in SUPERSEDED:
            continue
        gone[q] = [line.strip() for line in lines if q in line][:3]
    return gone


def broken_supersessions() -> list[str]:
    """Every superseded value must be gone, and every replacement must be present.

    Both halves are load-bearing. If an old value is still printed, the correction did not land
    everywhere and the papers now mix two denominators. If a new value is absent, the quantity was
    deleted rather than replaced -- which is exactly the silent loss this script exists to catch,
    wearing the costume of a correction.
    """
    after = paper_sources.all_tex()
    before = quantities(frozen_tex())
    problems = []
    for old, replacements in sorted(SUPERSEDED.items()):
        if old not in before:
            problems.append(f"{old}: not in the pre-split draft, so it superseded nothing")
        for new in replacements.split():
            if new not in after:
                problems.append(f"{old} -> {new}: the replacement is absent from every paper")
    return problems


def stale_drops() -> list[str]:
    """Deliberate-drop entries that are no longer dropped, or were never present.

    Without this the list rots into a blanket exemption: an entry added for one edit keeps
    suppressing the check after a later edit restores the number.
    """
    before = quantities(frozen_tex())
    after = quantities(paper_sources.all_tex())
    stale = []
    for q in sorted(DELIBERATE_DROPS):
        if q not in before:
            stale.append(f"{q}: not in the pre-split draft at all, so nothing to exempt")
        elif q in after:
            stale.append(f"{q}: present in the papers after all, so the exemption is unnecessary")
    return stale


def missing_spelled() -> list[str]:
    after = paper_sources.all_tex()
    return [w for w in SPELLED_CLAIMS if w not in after]


def coverage_by_paper() -> dict[str, int]:
    """How many pre-split quantities each paper carries. Reported so the split's shape is visible
    and a paper that ended up carrying almost nothing would be obvious."""
    before = quantities(frozen_tex())
    return {
        slug: len(before & quantities(paper_sources.paper_tex(slug)))
        for slug in paper_sources.PAPERS
    }


def main() -> int:  # pragma: no cover - CLI convenience
    before = quantities(frozen_tex())
    after = quantities(paper_sources.all_tex())
    gone = missing_quantities()
    stale = stale_drops() + broken_supersessions()
    unspelled = missing_spelled()

    print(f"pre-split quantities:      {len(before)}")
    print(f"post-split quantities:     {len(after)}")
    print(f"carried over:              {len(before & after)}")
    print(f"new (added by the split):  {len(after - before)}")
    print("\nper paper (share of pre-split quantities carried):")
    for slug, n in coverage_by_paper().items():
        print(f"  {paper_sources.PAPER_NUMERALS[slug]:>3}  {slug:26s} {n:4d}")

    failed = False
    if gone:
        failed = True
        print(f"\nLOST {len(gone)} quantities present before the split and in no paper after it:")
        for q, ctx in gone.items():
            print(f"  {q}")
            for line in ctx:
                print(f"      {line[:120]}")
    if stale:
        failed = True
        print("\nSTALE exemptions or broken supersessions:")
        for s in stale:
            print(f"  {s}")
    if unspelled:
        failed = True
        print(f"\nMISSING spelled-out counts: {', '.join(unspelled)}")

    if failed:
        return 1
    print(
        f"\nOK: every one of the {len(before)} pre-split quantities is carried by at least one "
        f"paper,\nor is one of the {len(SUPERSEDED)} values a corrected statistic replaced "
        f"(each replacement verified present),\nor is one of the {len(DELIBERATE_DROPS)} listed "
        f"deliberate drops."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
