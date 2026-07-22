"""Deterministic record/replay for LLM calls (VCR-lite, pure stdlib).

The validation study drives two live LLM surfaces: the worker `complete(prompt)`
and gepa's `reflection_lm(messages, **kwargs)`. Both are plain `input -> str`
callables, so both can be wrapped by a `Cassette`: one `record` run captures
every distinct call to a JSON file; every later run `replay`s from that file,
spending zero tokens (CI never touches the network).

Keyed by a stable sha256 of ``(name, args, kwargs)`` — not by call order — so a
replay survives reordering of arms or seeds between record and replay runs.
`record` mode dedups: an identical call is served from the cache instead of
re-hitting the model, which also lowers the cost of the one paid run.

Depends only on the standard library, so it imports and tests without the
`whystuck[eval]` extra (no gepa, no litellm).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

Mode = Literal["record", "replay"]

_VERSION = 1


class CassetteMiss(Exception):
    """Raised in replay mode when a call has no recorded output."""


def _key(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    # ensure_ascii + sort_keys make the serialization stable across runs;
    # default=str lets non-JSON inputs (rare) hash deterministically rather
    # than raise. The name namespaces surfaces so worker and reflection calls
    # with identical payloads never collide.
    canonical = json.dumps(
        {"name": name, "args": list(args), "kwargs": kwargs},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


class Cassette:
    """A JSON-backed store of LLM call outcomes, used as a context manager."""

    def __init__(self, path: str | Path, mode: Mode) -> None:
        self.path = Path(path)
        self.mode = mode
        self._entries: dict[str, dict[str, Any]] = {}
        if self.mode == "replay":
            self._load()
        elif self.path.exists():
            # Resume an interrupted recording rather than clobber it.
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            if self.mode == "replay":
                raise CassetteMiss(f"no cassette at {self.path}")
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = data.get("entries", {})

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _VERSION, "entries": self._entries}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    def wrap(self, fn: Callable[..., str], *, name: str) -> Callable[..., str]:
        """Wrap a `input -> str` model callable in record/replay behavior."""

        def wrapped(*args: Any, **kwargs: Any) -> str:
            key = _key(name, args, kwargs)
            entry = self._entries.get(key)
            if entry is not None:
                return str(entry["output"])
            if self.mode == "replay":
                raise CassetteMiss(f"no recorded output for {name}(*{args!r}, **{kwargs!r})")
            output = str(fn(*args, **kwargs))
            self._entries[key] = {"name": name, "output": output}
            self._persist()
            return output

        return wrapped

    def __enter__(self) -> Cassette:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.mode == "record":
            self._persist()
