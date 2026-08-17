"""A number the three papers share must be the same number in all of them.

`verify_paper_claims._SHARED` already checks this, for eleven patterns somebody thought to write
down. That list is the whole coverage, and the duplication it guards is much larger than the list:
about three hundred words run verbatim between Paper I's threats section and Paper III's section on
sigma, another two hundred and ninety between Paper I's and Paper II's related work, and the same
external results are quoted with their percentages in two papers at a time. Correcting one copy and
not the other is the defect this series is about, and for anything outside those eleven patterns
nothing was watching for it.

So this sweeps instead of listing. Any sentence that appears in two papers in near-identical words
has to carry identical numbers, and any cited external result quoted with a percentage in more than
one paper has to be quoted with the same percentages. Neither test names a sentence or a citation,
so new duplication is covered the moment it is written rather than when someone remembers to add a
pattern.

The sweep's own yield is asserted, because a sweep that stops matching reports agreement rather
than failure, and `test_the_sweep_catches_a_number_changed_in_one_copy` proves the comparison bites
on prose built for the purpose.
"""

from __future__ import annotations

import difflib
import itertools
import pathlib
import re
import sys
from functools import cache

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import paper_sources  # noqa: E402

#: A decimal or an integer, as the papers write them. Signs and `$` are excluded on purpose: the
#: comparison is between two copies of one sentence, so a sign that differs shows up as a word
#: difference in the similarity score rather than as a number difference here.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: Words per sentence below which a match means nothing: "so it is $3$." matches a hundred others.
_MIN_WORDS = 12

#: A full stop, and nothing else. Splitting on colons and semicolons too was the first version, and
#: it cut two of the four shared citations off from the percentage they attribute -- the sentence
#: that quotes OPRO and EvoPrompt carries the numbers on the far side of a semicolon. A unit that
#: ends at the nearest punctuation mark reads more cleanly and guards less.
_SENTENCE_END = re.compile(r"(?<=\.)\s")

#: How alike two sentences must read, with their numbers masked out, to count as the same sentence
#: restated. Measured on masked text so that a pair which differs ONLY in its numbers still scores
#: 1.0 and is compared -- that pair is the whole point.
_SIMILARITY = 0.85

#: A floor on how much duplication the sweep still sees. Not a measurement: the count moves with
#: every edit to either paper (it was 63 when this was written). It exists so that a change which
#: puts the sweep out of reach of the papers' prose -- a different sentence terminator, a wrapper
#: macro around every restatement -- fails here instead of quietly passing on nothing.
_MIN_PAIRS = 20

#: Table scaffolding, which is not prose. `p{0.20\textwidth}` and `\setlength{\tabcolsep}{5pt}`
#: carry digits that are layout, and two tables built the same way match each other perfectly. The
#: cells themselves are checked cell-by-cell by `verify_paper_claims`; captions stay in scope here.
_SCAFFOLD = re.compile(
    r"\\begin\{tabular\}.*?\\end\{tabular\}|\\setlength\{[^}]*\}\{[^}]*\}"
    r"|\\(?:begin|end)\{[^}]*\}|\\(?:top|mid|bottom)rule|\\centering|\\small|\\label\{[^}]*\}",
    re.DOTALL,
)


def _prose(slug: str) -> str:
    tex = paper_sources.paper_tex(slug)
    tex = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%"))
    return re.sub(r"\s+", " ", _SCAFFOLD.sub(" ", tex))


@cache
def _sentences(slug: str) -> tuple[str, ...]:
    """Sentences of `slug` that are long enough to match uniquely and carry at least one number."""
    return tuple(
        s.strip()
        for s in re.split(_SENTENCE_END, _prose(slug))
        if len(s.split()) >= _MIN_WORDS and _NUMBER.search(s)
    )


def _mask(sentence: str) -> str:
    return _NUMBER.sub("#", sentence)


def _numbers(sentence: str) -> list[str]:
    """Every number in the sentence, sorted. A multiset: a sentence may repeat one."""
    return sorted(_NUMBER.findall(sentence))


def _shingles(sentence: str) -> set[str]:
    words = _mask(sentence).lower().split()
    return {" ".join(words[i : i + 6]) for i in range(max(1, len(words) - 5))}


#: How many six-word shingles two sentences must share before their similarity is worth computing.
#: One shared shingle is met by boilerplate ("as a share of the split it does not"), which put
#: thousands of pairs through `difflib.ratio` and cost this file seven minutes of a pre-push gate --
#: the sort of slow that gets a hook switched off. Two is enough of a filter to make it seconds and
#: cannot lose a pair: a match at 0.85 over twelve words shares far more than two shingles.
_MIN_SHARED_SHINGLES = 2


def _restatements(left: tuple[str, ...], right: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """Sentence pairs that read the same with their numbers masked out.

    Shingled first, then filtered by shared-shingle count, then by `quick_ratio`, so the expensive
    comparison runs on a handful of pairs: comparing every sentence against every other is 380x211
    ratio computations over long word lists for one pairing alone.
    """
    index: dict[str, set[int]] = {}
    masked_right = [_mask(sentence).split() for sentence in right]
    for j, sentence in enumerate(right):
        for shingle in _shingles(sentence):
            index.setdefault(shingle, set()).add(j)
    pairs = []
    for sentence in left:
        hits: dict[int, int] = {}
        for shingle in _shingles(sentence):
            for j in index.get(shingle, ()):
                hits[j] = hits.get(j, 0) + 1
        words = _mask(sentence).split()
        for j, shared in hits.items():
            if shared < _MIN_SHARED_SHINGLES:
                continue
            matcher = difflib.SequenceMatcher(None, words, masked_right[j])
            # An upper bound on the real ratio, computed from a multiset intersection. Cheap enough
            # to run on every candidate and exact enough to discard almost all of them.
            if matcher.quick_ratio() < _SIMILARITY:
                continue
            ratio = matcher.ratio()
            if ratio >= _SIMILARITY:
                pairs.append((sentence, right[j], ratio))
    return pairs


@cache
def _all_restatements() -> list[tuple[str, str, str, str, float]]:
    out = []
    for a, b in itertools.combinations(paper_sources.PAPERS, 2):
        for left, right, ratio in _restatements(_sentences(a), _sentences(b)):
            out.append((a, b, left, right, ratio))
    return out


def test_the_sweep_still_finds_the_duplicated_prose():
    """The tripwire. Agreement over an empty set is not agreement."""
    pairs = _all_restatements()
    assert len(pairs) >= _MIN_PAIRS, (
        f"the sweep found {len(pairs)} restated sentence pairs, under the floor of {_MIN_PAIRS}. "
        "Either the papers stopped restating each other, or this sweep stopped being able to see "
        "it -- and the second one looks exactly like agreement"
    )


def test_near_duplicate_sentences_carry_the_same_numbers():
    for a, b, left, right, ratio in _all_restatements():
        assert _numbers(left) == _numbers(right), (
            f"{a} and {b} restate one sentence at similarity {ratio:.2f} with different numbers:\n"
            f"  {paper_sources.PAPER_NUMERALS[a]}: {left[:300]}\n"
            f"  {paper_sources.PAPER_NUMERALS[b]}: {right[:300]}\n"
            f"  {_numbers(left)} against {_numbers(right)}"
        )


def test_the_sweep_catches_a_number_changed_in_one_copy():
    """The mutation this guard exists for, on prose written to be mutated.

    Built rather than borrowed from the papers: a test that edits a real sentence would pass on
    the day someone rewords that sentence, which is the failure mode of every check in this
    repository that pinned itself to one site.
    """
    original = (
        "Every interval in this study contains within-condition sampling variance, because the "
        "worker runs at the provider's default temperature with one sample per task, and sigma "
        "for one task under one draw is 0.1125 on a thirty-task mean."
    )
    corrected = original.replace("0.1125", "0.1591")
    pairs = _restatements((original,), (corrected,))
    assert pairs, "the two copies no longer read as the same sentence, so nothing was compared"
    left, right, _ratio = pairs[0]
    assert _numbers(left) != _numbers(right), "a number changed in one copy went unnoticed"
    # And the same pair with the number left alone must not be flagged, or the guard would fail on
    # every duplicated sentence and be turned off within a week.
    same = _restatements((original,), (original,))
    assert same and _numbers(same[0][0]) == _numbers(same[0][1])


#: How the papers cite. `\citep[see][]{key}` and `\citet{a,b}` both appear.
_CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")

#: A percentage as the papers print it: `$6\%$`, `${\sim}20\%$`.
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\\%")

#: Cited works quoted with a percentage in more than one paper. Four today (GEPA, OPRO, EvoPrompt,
#: Self-Refine), and a floor for the same reason as `_MIN_PAIRS`: these are external results, so a
#: wrong one is a misattribution rather than an arithmetic slip, and nothing else re-derives them.
_MIN_SHARED_CITATIONS = 4


def _percentages_by_citation() -> dict[str, dict[str, set[str]]]:
    """Citation key -> paper -> the percentages quoted in the same sentence as that key."""
    out: dict[str, dict[str, set[str]]] = {}
    for slug in paper_sources.PAPERS:
        for sentence in re.split(_SENTENCE_END, _prose(slug)):
            found = _PERCENT.findall(sentence)
            if not found:
                continue
            for group in _CITE.findall(sentence):
                for key in (k.strip() for k in group.split(",") if k.strip()):
                    out.setdefault(key, {}).setdefault(slug, set()).update(found)
    return out


def test_a_cited_external_result_is_quoted_identically_in_every_paper():
    """Three related-work sections quote the same papers, and the numbers were copied by hand.

    Two hundred and eighty-nine words of Paper I's related work also stand in Paper II's, and a
    hundred and seven in Paper III's, including the percentages this study attributes to GEPA,
    OPRO, EvoPrompt and Self-Refine. Nothing derives those, since they belong to other authors; the
    most a checker can do is refuse to let two of our papers attribute different numbers to one
    result.
    """
    by_citation = _percentages_by_citation()
    shared = {key: byp for key, byp in by_citation.items() if len(byp) > 1}
    assert len(shared) >= _MIN_SHARED_CITATIONS, (
        f"only {len(shared)} cited works are quoted with a percentage in more than one paper, "
        f"under the floor of {_MIN_SHARED_CITATIONS}; the citation or percentage pattern has "
        "probably stopped matching the papers' own markup"
    )
    for key, by_paper in sorted(shared.items()):
        quoted = {slug: sorted(values) for slug, values in by_paper.items()}
        assert len({tuple(v) for v in quoted.values()}) == 1, (
            f"the papers attribute different percentages to {key}: "
            + ", ".join(
                f"{paper_sources.PAPER_NUMERALS[slug]}={values}"
                for slug, values in sorted(quoted.items())
            )
        )
