# What the accepted baseline contains, and why

`papyrustrail accept` writes the current findings as the accepted baseline. Its docstring calls
that "a reviewed action, not a gate", and the first baseline in this repo was not reviewed at all:
704 findings went in unexamined. This file records the review, so that anyone reading
`baseline.json` can tell an accepted gap from an accepted defect.

No paper was edited. Every reduction below came from fixing the checker or from declaring a
derivation the config had never expressed.

## The papers were not wrong. The tool and the config were.

| stage | bind findings | what changed |
|---|---|---|
| first run | 704 | nothing yet |
| identifier fix | 657 | 47 findings were hyphenated names (`gpt-4.1-mini`, `llama-3.3-70b`, `Apache-2.0`, `MATH-500`) whose tails parsed as negative measurements |
| `contrasts` block | 520 | 137 findings, including all 32 copies of the headline L1 loss, were one undeclared derivation |

The headline case is worth stating plainly, because it looked like a defect and was not. The papers
print the L1 loss as -0.042 in 32 places. No spec could express where it comes from, which is
`mean_i(runs[i].D.delta - runs[i].B.delta)` over the three seeds of one study. Recomputed from that
study: -0.095, -0.080, +0.050, mean -0.041666..., seed mean 0.78833. Every figure in the paper's row
matches, including the 0.7883 column and the 2/3 count. The paper was right and the registry was
silent, which is the failure mode a base rate of p = 0.017 predicts: most findings are gaps in your
own declaration, and bulk-accepting them buries the few that are not.

## What the remaining 520 are

426 of the 520 sites repeat one of about 102 quantities, so the tail is thin. Ranked by blast radius:

**The noise floor and everything divided by it (40 sites).** The papers define
`sigma(dd) = 0.1591/sqrt(30) = 0.0291` and report the L1 cell as `1.43 sigma`. Three values, 21 + 6
+ 13 sites. `0.1591` is a pooled sd and `0.0291` is its standard error, both shapes `contrasts`
already emits, so declaring them is mechanical ONCE someone says which 30 observations they pool.
Four candidate pools were tried and none reproduces 0.1591: the sweep's D-B differences (n = 87,
sd = 0.0411), every arm's `seed_heldout` (n = 471, sd = 0.1124), `D.delta` alone (n = 87,
sd = 0.0505), and per-task D-B differences within a level (n = 150, sd = 0.1779 / 0.1542 / 0.0963).
This was recorded here as a genuine open question, "40 published sites rest on a number no
declared artifact reproduces". That was wrong, and correcting it matters because the sentence read
as a defect in the papers. The papers' own checker derives the value: `tex: noise sigma delta task`
compares the printed `0.1591` against a computed `0.1591246970504689` and passes. Four guessed
pools failed to reproduce it because the guesses were wrong, not because nothing does. What is
open is a papyrustrail CONFIG gap, not a paper claim: the pool is known to
`scripts/verify_paper_claims.py` and has not been transcribed into a `contrasts` block.

**Magnitude printed for a signed artifact (13 sites).** The papers write `0.042` where the artifact
holds -0.041666. In prose "a loss of 0.042" is correct English, and `bind` requires an exact sign on
purpose, because a claim that a treatment helped when the artifact says it hurt is the defect the
rule exists for. Both readings are defensible and the rule stays as it is; these sites are accepted
as a known cost of it rather than silently excused.

**Ratios of two derived quantities (the `1.43` family).** `1.43` is `0.041666/0.029052`, a mean over
one study divided by a noise floor pooled over everything else. `contrasts` emits `t` as
`mean/se` from a single spec's own draws, so it cannot express a ratio whose numerator and
denominator come from different sources. Expressing it needs a fourth derivation kind, and that is
not worth building until the denominator above is pinned down.

**The long tail (94 sites, one occurrence each).** Individually cheap, collectively the least
informative: a quantity printed once is usually a derived figure in prose. Left accepted.

## cite

16 findings, 14 numerals attributed to nobody and 2 numeric claims citing sources whose notes
record no numbers. This paragraph said 22 and 8 until now, which were the counts before `cite` was
narrowed: it had been firing on any paragraph that cited a source beside our own measurements, so
citing `lakens2018equivalence` for what a SESOI means next to our own 0.05 read as an uncheckable
numeric claim about that source. Six of the eight went that way. The two that remain each flag a
ratio of two derived quantities, which is the registry's gap rather than the bibliography's.

Only 4 of the 29 bib entries record a numeral at all, so 25 citations have nothing for `cite` to
compare against in either direction. Three of the corpus's twenty defects were a paper misreporting
a source, and all three were found by a person fetching the source. Nothing here reaches that.

## The two sites accepted on 2026-09-03

`bind` gained two sites, both `-3.3`, one in each of Paper~I and Paper~III, in the sentence that
reports what \citet{bai2026masprompt} measured. They are accepted for the same reason the
`+4.2`, `-16.0` and `+24.0` beside them already were: the number belongs to the cited paper, and
no artifact of ours can reproduce someone else's table. What changed is that the sentence used to
repeat that paper's own prose claim of improvement "across all five diverse tasks" while quoting
an average taken over nine, and its nine-task single-agent column is flat on LiveCodeBench and
`-3.3` on SWE-Bench Verified. Reporting the two exceptions is what added the numeral. The value
was read off the source PDF, and the bibliography note records where, so a future reader can
check it without refetching.
