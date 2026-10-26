"""Every Groq model we are willing to spend on must be a model Groq actually serves.

WHY THIS EXISTS. `MODEL_PRICES` is the pre-flight spend gate: an unpriced model is refused
before any call. That makes the table a whitelist, and a whitelist is only as good as its
entries. It carried `groq/openai-gpt-oss-120b` from 2026-08-06 to 2026-08-08, which is not a
model id -- Groq namespaces that one `openai/gpt-oss-120b`, so the litellm string needs two
slashes. The gate would have ADMITTED a run naming it and the run would have died on its first
call, after the estimate printed and the operator had said yes.

That is the same shape as the defects these papers are about: a check that reports OK about a
thing it never verified. Here the fix is cheap, because Groq publishes the list.

The list is a dated snapshot rather than a live call, so the suite stays offline and $0. A
snapshot goes stale, which is the honest trade: it cannot catch a model Groq retires tomorrow,
but it does catch a typo or a wrong namespace, which is the failure that actually happened.
Refresh it by re-fetching `/v1/models`; the date in the filename is the claim about when it was
true.
"""

from __future__ import annotations

import json
import pathlib

from whystuck.eval.cost import MODEL_PRICES

SNAPSHOT = pathlib.Path(__file__).parent / "fixtures" / "groq-models-2026-08-08.json"
#: Groq retires models. By 2026-10-25 both llama workers were gone, while the released runs
#: that name them still need their prices. So a priced model must have been served on SOME
#: dated snapshot; each snapshot remains a claim about its own date only.
SNAPSHOTS = sorted(pathlib.Path(__file__).parent.joinpath("fixtures").glob("groq-models-*.json"))


def _ids(path: pathlib.Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8"))["active_model_ids"])


def _served() -> set[str]:
    return _ids(SNAPSHOT)


def _served_ever() -> set[str]:
    return set().union(*(_ids(p) for p in SNAPSHOTS))


def test_every_priced_groq_model_is_one_groq_serves():
    served = _served_ever()
    priced = {m for m in MODEL_PRICES if m.startswith("groq/")}
    assert priced, "no Groq models priced; delete this guard or restore the entries"
    unknown = sorted(m for m in priced if m.removeprefix("groq/") not in served)
    assert not unknown, (
        f"priced but not served by Groq: {unknown}. The pre-flight gate would admit a run "
        f"naming one of these and it would 404 on its first call. Served ids: {sorted(served)}"
    )


def test_the_snapshot_is_not_empty_so_the_guard_cannot_pass_vacuously():
    # An empty or malformed snapshot makes `served` empty, and then EVERY priced model looks
    # unknown -- loud, not silent. The real risk is the reverse: a snapshot that somehow
    # contains everything would make the guard meaningless, so assert it is a real list.
    served = _served()
    assert 5 < len(served) < 500
    assert "llama-3.3-70b-versatile" in served, "the D3 critic must be in any honest snapshot"


def test_a_wrong_namespace_is_reported():
    # The exact bug: `openai/gpt-oss-120b` is served, `openai-gpt-oss-120b` is not.
    served = _served()
    assert "openai/gpt-oss-120b" in served
    assert "openai-gpt-oss-120b" not in served


def test_every_snapshot_is_a_real_list():
    assert len(SNAPSHOTS) >= 2
    for path in SNAPSHOTS:
        ids = _ids(path)
        assert 5 < len(ids) < 500, path
        assert "openai/gpt-oss-120b" in ids, path


def test_a_typo_is_still_reported_against_the_union():
    # The union must not launder a namespace typo just because it is wider.
    assert "openai-gpt-oss-120b" not in _served_ever()
    assert "qwen3.8-27b" not in _served_ever()
