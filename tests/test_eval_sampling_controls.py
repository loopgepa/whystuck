"""Temperature and sample-salt controls, and the cassette-key property they depend on.

D7's premise: no interval in this study contains within-condition sampling variance, because
the worker draws one sample per task at the provider's default temperature. Fixing that needs
a pinned temperature and repeated samples -- and both are traps for the cassette, which keys
on (name, args, kwargs) and DEDUPES identical calls in record mode.

- A temperature baked into the `litellm_complete` factory would not appear in the key, so two
  runs at different temperatures would serve each other's completions. It is therefore passed
  per call.
- K identical sampling calls would collapse to one cassette entry and return byte-identical
  "samples", which reads as zero sampling variance -- the most flattering possible artifact.
  A `sample` salt keeps the keys distinct without being sent to the provider.

Both default to None so every committed cassette key is preserved; that is asserted here.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from whystuck.eval.cassette import Cassette


def _recorded_keys(path: pathlib.Path) -> set[str]:
    return set(json.loads(path.read_text())["entries"])


def test_temperature_changes_the_cassette_key(tmp_path):
    calls = []

    def fake(prompt, **kw):
        calls.append((prompt, kw))
        return f"out{len(calls)}"

    path = tmp_path / "c.json"
    with Cassette(str(path), mode="record") as c:
        w = c.wrap(fake, name="worker")
        a = w("p")
        b = w("p", temperature=0.0)
        again = w("p")
    assert a != b, "different temperature must not be served the same cached completion"
    assert again == a, "same call must still dedupe"
    assert len(calls) == 2
    assert len(_recorded_keys(path)) == 2


def test_sample_salt_changes_the_cassette_key(tmp_path):
    n = 0

    def fake(prompt, **kw):
        nonlocal n
        n += 1
        return f"draft{n}"

    with Cassette(str(tmp_path / "c.json"), mode="record") as c:
        w = c.wrap(fake, name="worker")
        outs = [w("p") if r == 0 else w("p", sample=r) for r in range(4)]
    assert len(set(outs)) == 4, "K samples collapsing to one entry would read as zero variance"
    assert n == 4


def test_unsalted_call_key_is_unchanged_by_the_new_kwargs(tmp_path):
    """The default path must hash exactly as it did before these parameters existed."""
    from whystuck.eval.cassette import _key

    assert _key("worker", ("p",), {}) == _key("worker", ("p",), {})
    assert _key("worker", ("p",), {}) != _key("worker", ("p",), {"temperature": 0.0})
    assert _key("worker", ("p",), {}) != _key("worker", ("p",), {"sample": 1})


def test_sample_is_a_salt_and_never_reaches_the_provider():
    """Sending `sample` to litellm would be an unknown parameter, and it is not a seed."""
    from whystuck.eval.llm_worker import litellm_complete

    seen = {}

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    import sys
    import types

    fake_litellm = types.ModuleType("litellm")

    def completion(**kw):
        seen.update(kw)
        return _Resp()

    fake_litellm.completion = completion  # type: ignore[attr-defined]
    sys.modules["litellm"] = fake_litellm
    try:
        out = litellm_complete("m", max_tokens=7)("p", sample=3, temperature=0.25)
    finally:
        del sys.modules["litellm"]
    assert out == "ok"
    assert "sample" not in seen
    assert seen["temperature"] == 0.25
    assert seen["max_tokens"] == 7


def test_temperature_omitted_entirely_when_unset():
    """Not `temperature=None` -- absent, so the request body matches the committed runs."""
    from whystuck.eval.llm_worker import litellm_complete

    seen = {}

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

    import sys
    import types

    fake_litellm = types.ModuleType("litellm")

    def completion(**kw):
        seen.update(kw)
        return _Resp()

    fake_litellm.completion = completion  # type: ignore[attr-defined]
    sys.modules["litellm"] = fake_litellm
    try:
        litellm_complete("m", max_tokens=7)("p")
    finally:
        del sys.modules["litellm"]
    assert "temperature" not in seen


def test_cli_exposes_temperature_and_defaults_to_unpinned():
    import importlib.util

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"
    spec = importlib.util.spec_from_file_location("run_validation_study", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._parse_args([]).temperature is None
    assert mod._parse_args(["--temperature", "0.0"]).temperature == pytest.approx(0.0)
