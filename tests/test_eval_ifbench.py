"""Tests for whystuck.eval.ifbench — vendored IFBench loader + programmatic scorer.

Verifiers are faithful ports of IFBench's check_following (Apache-2.0); these
tests pin their behavior on constructed strings plus the loader's filtering and
deterministic splitting. Fully offline and deterministic.
"""

from __future__ import annotations

import pytest

from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import (
    SUPPORTED_IDS,
    VERIFIERS,
    ifbench_pass,
    ifbench_score,
    load_ifbench,
)


def _task(ids, prompt="p", key="k"):
    return BenchmarkTask(id=key, payload={"goal": prompt, "instruction_id_list": list(ids)})


# --- verifier fidelity (representative pass/fail per constraint) ---------------


@pytest.mark.parametrize(
    "iid,good,bad",
    [
        # all 11 verifiers get one true-positive and one true-negative; strings
        # verified against the (faithful-port) implementations.
        ("format:no_whitespace", "nowhitespaceatall", "has space"),
        ("format:parentheses", "text (((((deep))))) more", "shallow ()"),
        ("custom:multiples", "14, 21, 28, 35, 42, 49", "1, 2, 3"),
        ("words:vowel", "hello world", "sequoia beautiful"),
        ("format:newline", "one\ntwo\nthree", "one two three"),
        ("format:sub-bullets", "* top\n- sub", "* top\n* another"),
        ("format:quotes", '"\'""\'"', 'She said "hello" once.'),
        (
            "count:punctuation",
            "Well?! Wait, really? Are you sure! Yes; go on: now.",
            "no combo full stop.",
        ),
        (
            "words:palindrome",
            "level radar civic rotor refer kayak redder reviver repaper rotator",
            "level radar civic",
        ),
        ("words:consonants", "black stops", "cat dog"),
        ("format:quote_unquote", 'He said "hello" and left', 'He said "hello"'),
    ],
)
def test_verifier_pass_and_fail(iid, good, bad):
    verify = VERIFIERS[iid]
    assert verify(good) is True
    assert verify(bad) is False


def test_quote_unquote_rejects_adjacent_double_quotes():
    # exercises the explicit `if '""' in value: return False` branch
    assert VERIFIERS["format:quote_unquote"]('a""b') is False


def test_quote_unquote_is_faithful_on_typographic_quotes():
    # Regression: an earlier port added curly-quote normalization upstream does
    # NOT do (its .replace('"','"') calls are ASCII no-ops), which wrongly failed
    # smart-quote-terminated outputs. Upstream returns True here — so must we.
    assert VERIFIERS["format:quote_unquote"]("this ends with a quote”") is True


def test_every_supported_id_has_a_verifier():
    assert frozenset(VERIFIERS) == SUPPORTED_IDS
    assert len(VERIFIERS) == 11


# --- scoring ------------------------------------------------------------------


def test_pass_requires_all_constraints():
    task = _task(["format:no_whitespace", "format:newline"])
    # "a\nb" -> no_whitespace False (newline is whitespace); not a full pass
    assert ifbench_pass("a\nb", task) is False


def test_score_is_fraction_of_constraints_met():
    task = _task(["format:no_whitespace", "custom:multiples"])
    # no_whitespace fails (has spaces), multiples passes -> 1/2
    assert ifbench_score("14, 21, 28, 35, 42, 49", task) == 0.5


def test_score_all_met_is_one_and_pass_true():
    task = _task(["format:no_whitespace"])
    assert ifbench_score("solid", task) == 1.0
    assert ifbench_pass("solid", task) is True


def test_no_constraints_scores_zero_and_never_passes():
    # An unscored row must not count as a free win (not vacuous 1.0/True).
    task = _task([])
    assert ifbench_score("anything", task) == 0.0
    assert ifbench_pass("anything", task) is False


def test_unknown_constraint_id_raises():
    # Fail loud: load_ifbench only emits supported ids, so an unknown id is a
    # caller bug, not something to silently skip and inflate the score.
    task = _task(["nope:unsupported"])
    with pytest.raises(KeyError):
        ifbench_score("x", task)
    with pytest.raises(KeyError):
        ifbench_pass("x", task)


def test_empty_output_trivially_satisfies_no_whitespace():
    # Pinned deliberately: an empty string has no whitespace, so it passes.
    assert ifbench_pass("", _task(["format:no_whitespace"])) is True


# --- loader -------------------------------------------------------------------


def test_load_returns_only_supported_rows():
    tasks = load_ifbench(split="all")
    assert len(tasks) == 62
    for t in tasks:
        assert set(t.payload["instruction_id_list"]) <= SUPPORTED_IDS
        assert t.payload["goal"]  # the IFBench prompt is the worker goal


def test_split_is_disjoint_and_covers_all():
    train = {t.id for t in load_ifbench(split="train")}
    held = {t.id for t in load_ifbench(split="heldout")}
    allids = {t.id for t in load_ifbench(split="all")}
    assert train and held
    assert train.isdisjoint(held)
    assert train | held == allids


def test_split_is_deterministic_per_seed():
    a = [t.id for t in load_ifbench(split="train", seed=7)]
    b = [t.id for t in load_ifbench(split="train", seed=7)]
    c = [t.id for t in load_ifbench(split="train", seed=8)]
    assert a == b  # same seed -> same order
    assert a != c  # different seed -> different shuffle


def test_n_limits_row_count():
    assert len(load_ifbench(split="train", n=5)) == 5


def test_n_larger_than_available_returns_all():
    # No error, no padding — just every available row.
    assert len(load_ifbench(split="all", n=10_000)) == 62
    assert len(load_ifbench(split="train", n=10_000)) == 31


def test_path_override_filters_unsupported_rows(tmp_path):
    # A custom file: one supported row, one unsupported (must be dropped).
    import json

    fixture = tmp_path / "custom.jsonl"
    fixture.write_text(
        json.dumps(
            {"key": "1", "prompt": "keep me", "instruction_id_list": ["format:no_whitespace"]}
        )
        + "\n"
        + json.dumps({"key": "2", "prompt": "drop me", "instruction_id_list": ["needs:nltk"]})
        + "\n",
        encoding="utf-8",
    )
    tasks = load_ifbench(split="all", path=fixture)
    assert [t.id for t in tasks] == ["1"]
    assert tasks[0].payload["goal"] == "keep me"


# NOTE: every row in the vendored slice is single-constraint, so ifbench_score's
# multi-constraint fraction path is covered via the synthetic _task() helper
# above rather than a loaded row — this is deliberate, not an oversight.
