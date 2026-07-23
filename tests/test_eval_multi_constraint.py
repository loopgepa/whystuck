"""Tests for whystuck.eval.multi_constraint — the composable multi-constraint set.

The single-constraint IFBench subset cannot form fair multi-constraint tasks (its
verifiers are trivially co-satisfied by a degenerate blob or mutually conflict).
This module instead uses canonical, INDEPENDENT instruction-following constraints
(keyword / length / case / ending / no-comma / bullets) with per-task parameters —
the families real IFBench/IFEval multi-constraint prompts use. Each constraint is
individually diagnosable and the set is jointly satisfiable by construction.

The load-bearing correctness property is SATISFIABILITY: every generated task must
be achievable (its witness scores 1.0), or the experiment floors every arm and the
result is meaningless. Most tests here defend that property.
"""

from __future__ import annotations

import pytest

from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.multi_constraint import (
    MC_VERIFIERS,
    build_witness,
    generate_multi_tasks,
    load_multi_constraint,
    mc_feedback,
    mc_pass,
    mc_score,
    render_multi_prompt,
)


def _task(pairs, goal="Write about a topic."):
    ids = [iid for iid, _ in pairs]
    kw = [k for _, k in pairs]
    return BenchmarkTask(id="t", payload={"goal": goal, "instruction_id_list": ids, "kwargs": kw})


# --- individual verifiers -----------------------------------------------------


def test_include_keyword_is_case_insensitive():
    v = MC_VERIFIERS["include_keyword"]
    assert v("the PHOTON travels", {"keyword": "photon"})
    assert not v("no match here", {"keyword": "photon"})


def test_min_words_counts_whitespace_tokens():
    v = MC_VERIFIERS["min_words"]
    assert v("one two three four five", {"n": 5})
    assert not v("one two three", {"n": 5})


def test_all_lowercase_rejects_any_uppercase_and_requires_letters():
    v = MC_VERIFIERS["all_lowercase"]
    assert v("all lower case words", {})
    assert not v("Has Uppercase", {})
    assert not v("12345 !!!", {})  # no letters -> not a lowercase-text


def test_end_with_matches_trailing_phrase_ignoring_trailing_space():
    v = MC_VERIFIERS["end_with"]
    assert v("… that is all.  ", {"phrase": "that is all."})
    assert not v("that is all. more", {"phrase": "that is all."})


def test_no_comma():
    v = MC_VERIFIERS["no_comma"]
    assert v("no commas here", {})
    assert not v("one, two", {})


def test_num_bullets_counts_bullet_lines_exactly():
    v = MC_VERIFIERS["num_bullets"]
    text = "- first\n- second\n- third"
    assert v(text, {"n": 3})
    assert not v(text, {"n": 4})
    assert not v("- only one", {"n": 3})


def test_exact_word_count_counts_whitespace_words_not_markers():
    v = MC_VERIFIERS["exact_word_count"]
    assert v("- one two\n- three four", {"n": 4})  # bullet markers "-" are not words
    assert not v("one two three", {"n": 4})
    assert v("don't stop now", {"n": 3})  # a contraction is ONE word (human sense)
    assert v("a well-known idea", {"n": 3})  # a hyphenated compound is ONE word


def test_forbidden_word_is_whole_word_and_not_tripped_by_hyphens():
    v = MC_VERIFIERS["forbidden_word"]
    assert not v("The cat sat", {"word": "the"})  # standalone word (any case) -> fails
    assert v("theory of forms", {"word": "the"})  # substring inside 'theory' is fine
    assert v("a state-of-the-art model", {"word": "the"})  # hyphenated idiom is NOT the word
    assert v("a clean sentence", {"word": "the"})


def test_word_max_length_bounds_each_letter_run():
    v = MC_VERIFIERS["word_max_length"]
    assert v("all short here", {"m": 5})
    assert not v("short longword here", {"m": 5})  # 'longword' is 8 letters
    assert v("state-of-the-art", {"m": 5})  # judged by letter-runs (state/of/the/art)


# --- scoring ------------------------------------------------------------------


def test_mc_score_is_fraction_satisfied():
    pairs = [
        ("include_keyword", {"keyword": "photon"}),
        ("no_comma", {}),
        ("min_words", {"n": 100}),
    ]
    # output has the keyword, no comma, but far fewer than 100 words -> 2/3
    assert mc_score("the photon moves fast", _task(pairs)) == pytest.approx(2 / 3)


def test_mc_score_empty_is_zero_not_vacuous_one():
    assert mc_score("anything", _task([])) == 0.0


def test_mc_pass_requires_all():
    pairs = [("no_comma", {}), ("include_keyword", {"keyword": "x"})]
    assert mc_pass("has x no comma", _task(pairs))
    assert not mc_pass("has x, comma", _task(pairs))


def test_mc_score_unknown_id_raises():
    t = BenchmarkTask(id="t", payload={"instruction_id_list": ["nope"], "kwargs": [{}]})
    with pytest.raises(KeyError):
        mc_score("x", t)


def test_mismatched_ids_and_kwargs_raise_with_task_id():
    # A malformed task (parallel lists of different lengths) must fail loud, naming
    # the task, not raise a context-free zip() error deep in scoring.
    t = BenchmarkTask(
        id="bad-42", payload={"instruction_id_list": ["no_comma", "min_words"], "kwargs": [{}]}
    )
    with pytest.raises(ValueError, match="bad-42"):
        mc_score("x", t)


# --- feedback + prompt --------------------------------------------------------


def test_mc_feedback_names_failed_constraint_with_its_parameter():
    pairs = [("include_keyword", {"keyword": "mitochondria"}), ("no_comma", {})]
    fb = mc_feedback("a cell, powerhouse", _task(pairs))
    assert "mitochondria" in fb  # the missing keyword is named
    assert "no_comma" in fb or "comma" in fb  # the comma failure is named


def test_render_multi_prompt_lists_every_requirement_and_the_goal():
    pairs = [
        ("include_keyword", {"keyword": "entropy"}),
        ("num_bullets", {"n": 3}),
        ("all_lowercase", {}),
    ]
    p = render_multi_prompt("Explain the idea.", pairs)
    assert "Explain the idea." in p
    assert "entropy" in p and "3" in p and "lowercase" in p


# --- SATISFIABILITY (the load-bearing property) -------------------------------


def test_build_witness_satisfies_every_selected_constraint():
    # A witness must pass ALL constraints in its set, for many combos.
    combos = [
        [("include_keyword", {"keyword": "photon"}), ("min_words", {"n": 20}), ("no_comma", {})],
        [("num_bullets", {"n": 4}), ("all_lowercase", {}), ("end_with", {"phrase": "the end."})],
        [
            ("include_keyword", {"keyword": "vector"}),
            ("num_bullets", {"n": 3}),
            ("min_words", {"n": 25}),
            ("no_comma", {}),
        ],
        [("all_lowercase", {}), ("no_comma", {}), ("end_with", {"phrase": "done."})],
    ]
    for pairs in combos:
        w = build_witness(pairs)
        t = _task(pairs)
        assert mc_score(w, t) == 1.0, f"witness failed for {[i for i, _ in pairs]}: {w!r}"


def _exhaustive_check(ids, grid, ks):
    from itertools import combinations, product

    checked = 0
    for k in ks:
        for combo in combinations(ids, k):
            for params in product(*(grid[c] for c in combo)):
                pairs = list(zip(combo, params, strict=True))
                assert mc_score(build_witness(pairs), _task(pairs)) == 1.0, (combo, params)
                checked += 1
    return checked


def test_easy_witness_satisfies_all_k3_k4_combos_exhaustively():
    # The load-bearing guarantee for the EASY tier, proven across the ENTIRE space the
    # generator can sample: every k=3,4 subset over every parameter value must have a
    # witness scoring exactly 1.0. A future edit that breaks build_witness fails HERE.
    from whystuck.eval.multi_constraint import (
        _EASY_IDS,
        _KEYWORDS,
        _MIN_WORDS,
        _NUM_BULLETS,
        _PHRASES,
    )

    grid = {
        "include_keyword": [{"keyword": k} for k in _KEYWORDS],
        "min_words": [{"n": n} for n in _MIN_WORDS],
        "end_with": [{"phrase": p} for p in _PHRASES],
        "num_bullets": [{"n": n} for n in _NUM_BULLETS],
        "all_lowercase": [{}],
        "no_comma": [{}],
    }
    assert _exhaustive_check(_EASY_IDS, grid, (3, 4)) > 3000


def test_hard_witness_satisfies_all_k4_k5_combos_exhaustively():
    # Same guarantee for the HARD tier — the higher-bug-risk one (exact counts,
    # forbidden word, max length interacting with bullets/phrase). Every k=4,5 subset
    # over every parameter value must witness to 1.0, or the paid run could floor arms.
    from whystuck.eval.multi_constraint import (
        _EXACT_WORDS,
        _FORBIDDEN,
        _HARD_IDS,
        _HARD_KEYWORDS,
        _HARD_PHRASES,
        _MAX_LEN,
        _NUM_BULLETS,
    )

    grid = {
        "include_keyword": [{"keyword": k} for k in _HARD_KEYWORDS],
        "exact_word_count": [{"n": n} for n in _EXACT_WORDS],
        "end_with": [{"phrase": p} for p in _HARD_PHRASES],
        "num_bullets": [{"n": n} for n in _NUM_BULLETS],
        "forbidden_word": [{"word": w} for w in _FORBIDDEN],
        "word_max_length": [{"m": m} for m in _MAX_LEN],
        "all_lowercase": [{}],
        "no_comma": [{}],
    }
    assert _exhaustive_check(_HARD_IDS, grid, (4, 5)) > 1000


def test_every_generated_task_is_satisfiable():
    # THE guarantee: the generator must never emit a task whose own witness can't
    # score 1.0. Without this, arms floor and the experiment is meaningless.
    tasks = generate_multi_tasks(30, seed=0)
    assert len(tasks) == 30
    for t in tasks:
        pairs = list(zip(t.payload["instruction_id_list"], t.payload["kwargs"], strict=True))
        assert 3 <= len(pairs) <= 4
        w = build_witness(pairs)
        assert mc_score(w, t) == 1.0


def test_hard_generated_tasks_are_satisfiable_with_two_plus_brutal():
    # The hard tier must honor satisfiability AND guarantee real difficulty: every
    # task carries exact_word_count (the skill-gap anchor) + at least one more brutal
    # constraint, so the seed can't re-saturate the way run 2's easy tier did.
    from whystuck.eval.multi_constraint import _BRUTAL

    tasks = generate_multi_tasks(30, seed=0, k_min=4, k_max=5, hard=True)
    for t in tasks:
        ids = t.payload["instruction_id_list"]
        pairs = list(zip(ids, t.payload["kwargs"], strict=True))
        assert 4 <= len(pairs) <= 5
        assert mc_score(build_witness(pairs), t) == 1.0
        assert "exact_word_count" in ids  # the anchor is always present
        assert len(set(_BRUTAL) & set(ids)) >= 2  # at least two hard constraints


def test_hard_tasks_penalize_a_competent_answer_more_than_easy():
    # Headroom proxy: the SAME fluent answer (has caps, a comma, long words, the word
    # "the", a natural word count) should satisfy a SMALLER fraction of hard tasks
    # than easy ones — evidence the hard tier leaves room below 1.0 for optimization.
    answer = (
        "The photon and the vector interact within this system, and the gradient "
        "updates the kernel as entropy evolves throughout the entire process."
    )
    easy = [mc_score(answer, t) for t in generate_multi_tasks(40, seed=9)]
    hard = [
        mc_score(answer, t) for t in generate_multi_tasks(40, seed=9, k_min=4, k_max=5, hard=True)
    ]
    assert sum(hard) / len(hard) < sum(easy) / len(easy)


def test_generated_tasks_carry_goal_and_parallel_kwargs():
    t = generate_multi_tasks(1, seed=1)[0]
    p = t.payload
    assert p["goal"]  # a rendered NL instruction
    assert len(p["instruction_id_list"]) == len(p["kwargs"])  # parallel lists
    # the goal text mentions the requirements (so the worker can see them)
    assert "requirement" in p["goal"].lower()


def test_generation_is_deterministic_by_seed():
    a = generate_multi_tasks(10, seed=7)
    b = generate_multi_tasks(10, seed=7)
    c = generate_multi_tasks(10, seed=8)
    assert [t.payload for t in a] == [t.payload for t in b]
    assert [t.payload for t in a] != [t.payload for t in c]


def test_load_split_is_disjoint_and_deterministic():
    train = load_multi_constraint("train", n=12, seed=0)
    heldout = load_multi_constraint("heldout", n=12, seed=0)
    train_ids = {t.id for t in train}
    heldout_ids = {t.id for t in heldout}
    assert train_ids and heldout_ids
    assert train_ids.isdisjoint(heldout_ids)  # no leakage between splits
    assert [t.payload for t in train] == [
        t.payload for t in load_multi_constraint("train", n=12, seed=0)
    ]


def test_load_raises_when_n_exceeds_available_half_pool():
    # Silent truncation would make meta["n_heldout"] disagree with the tasks actually
    # scored. Asking for more than the split can supply must raise, not quietly shrink.
    with pytest.raises(ValueError, match="pool"):
        load_multi_constraint("train", n=40, seed=0, pool=64)  # half-pool is 32 < 40
    # …and a big enough pool serves the request
    assert len(load_multi_constraint("train", n=40, seed=0, pool=100)) == 40


def test_competent_answer_scores_fractional_not_a_cliff():
    # A fluent but not constraint-obsessed answer (capitalized, a comma, real words)
    # partially satisfies tasks — fractional scores mean the loop has a gradient to
    # climb (headroom), not a 0/1 cliff, and nothing is trivially aced.
    answer = (
        "Photons and vectors interact in this system, and the gradient updates "
        "each epoch as the kernel and entropy evolve over time."
    )
    scores = [mc_score(answer, t) for t in generate_multi_tasks(40, seed=5)]
    assert any(0.0 < s < 1.0 for s in scores)  # graceful partial credit exists
    assert all(s < 1.0 for s in scores)  # nothing trivially aced (has caps + comma)


def test_tasks_have_headroom_seed_answer_is_not_trivially_perfect():
    # A naive answer (ignores the constraints) must NOT already score 1.0 — else
    # there's nothing for GEPA to optimize. Use a plain sentence as the naive answer.
    naive = "Here is a short answer, with a comma and Capitals."
    tasks = generate_multi_tasks(20, seed=3)
    scores = [mc_score(naive, t) for t in tasks]
    assert max(scores) < 1.0  # no task is trivially aced by the naive answer
