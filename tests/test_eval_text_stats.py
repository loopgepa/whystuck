"""Tests for whystuck.eval.text_stats — the non-oracle feedback tool.

`text_stats` is the "external tool" signal from the feedback-richness study
(Huang et al. 2023: intrinsic self-correction fails, but self-correction with a
real verification tool works). It reports MEASURED properties of a text and
nothing about the task's constraints — so it is a measurement, not the oracle
(which would return per-constraint pass/fail). These tests pin both properties:
the measurements are correct, and the function never sees the task.
"""

import inspect

from whystuck.eval import text_stats as ts
from whystuck.eval.multi_constraint import _tokens, _words


def test_reports_word_count_matching_the_scorer():
    # The word count it reports must equal what the exact_word_count verifier
    # measures (_tokens), or the "tool" would lie and misguide the refine step.
    out = "one two three four five"
    assert f"words={len(_tokens(out))}" in ts.text_stats(out)


def test_word_and_letter_tokenizers_agree_with_multi_constraint():
    # Guard against drift: text_stats must count words/letters the SAME way the
    # multi_constraint scorer does, else the feedback is unfaithful.
    for s in ["Hello, world!", "well-known don't x?!.", "- bullet\n- two", "UPPER lower"]:
        assert ts._word_tokens(s) == _tokens(s)
        assert ts._letter_words(s) == _words(s)


def test_reports_bullets_and_longest_word():
    out = "- alpha\n- betamax\nplain line, with comma"
    stats = ts.text_stats(out)
    assert "bullet_lines=2" in stats
    assert "longest_word_letters=7" in stats  # "betamax"


def test_omits_oracle_fields_all_lowercase_and_commas():
    # `all_lowercase` and `no_comma` verifiers have no per-task parameter, so a
    # measurement of either IS their pass/fail verdict — reporting it would be
    # oracle contamination (run-4 design review). It must never appear.
    stats = ts.text_stats("Has A Capital, and a comma")
    assert "all_lowercase" not in stats
    assert "commas" not in stats


def test_is_single_line_and_cheap():
    # Injected into every critique prompt, so it must stay one compact line.
    assert "\n" not in ts.text_stats("multi\nline\ninput text here")


def test_takes_only_the_output_string_never_the_task():
    # Structural proof it cannot be the oracle: its signature has exactly one
    # positional parameter (the text) and no task/constraint argument.
    params = list(inspect.signature(ts.text_stats).parameters.values())
    assert len(params) == 1
    assert params[0].name in {"output", "text"}
