"""WhyStuck local file store (ADR-004).

Layout under the store root (default outside the repo — see `resolve_store_root`):

    loops/<loop_id>.yaml
    traces/<loop_id>.jsonl          # append-only, one trace per line
    variants/<loop_id>/<variant_id>.yaml
    promotions/<loop_id>.jsonl      # append-only, one record per line

Genomes/variants are YAML (human-diffable); traces/promotions are JSONL so
concurrent runs can append a single line atomically (POSIX O_APPEND).
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from ruamel.yaml import YAML

from whystuck.core.errors import NotFoundError
from whystuck.core.models import (
    LoopGenome,
    LoopTrace,
    LoopVariant,
    PromotionRecord,
)

_yaml = YAML(typ="safe")
_yaml.default_flow_style = False


def resolve_store_root(env: Mapping[str, str]) -> Path:
    """Resolve the store root: WHYSTUCK_HOME > $XDG_DATA_HOME/whystuck >
    ~/.local/share/whystuck. Never the current working directory (ADR-004.2)."""
    if home := env.get("WHYSTUCK_HOME"):
        return Path(home)
    if xdg := env.get("XDG_DATA_HOME"):
        return Path(xdg) / "whystuck"
    home = env.get("HOME")
    root = Path(home) if home else Path("~").expanduser()
    return root / ".local" / "share" / "whystuck"


def _dump_yaml(data: object) -> str:
    buf = io.StringIO()
    _yaml.dump(data, buf)
    return buf.getvalue()


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte — os.write may return a short count."""
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a rename is durable across a crash."""
    try:
        dfd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a unique temp file (same dir → same filesystem) + fsync +
    os.replace so a reader never sees a partial file. The file bytes and the
    rename are both fsynced, so a completed write survives a crash. A unique temp
    name (mkstemp, mode 0o600) means concurrent writers to the same path never
    clobber each other's temp; the temp is removed on any error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        _write_all(fd, text.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _append_line(path: Path, text: str) -> None:
    """Append one line under O_APPEND. POSIX positions each write atomically at
    EOF (no lost appends); lines are small (output is capped) so the bytes are
    not interleaved on Linux. Raw os.write (looped for short counts), not buffered
    text I/O, keeps the append a single positioned write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (text + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        _write_all(fd, line)
    finally:
        os.close(fd)


class WhyStuckStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> WhyStuckStore:
        return cls(resolve_store_root(env if env is not None else os.environ))

    # --- paths ---
    def _loop_path(self, loop_id: str) -> Path:
        return self.root / "loops" / f"{loop_id}.yaml"

    def _traces_path(self, loop_id: str) -> Path:
        return self.root / "traces" / f"{loop_id}.jsonl"

    def _variant_path(self, loop_id: str, variant_id: str) -> Path:
        return self.root / "variants" / loop_id / f"{variant_id}.yaml"

    def _promotions_path(self, loop_id: str) -> Path:
        return self.root / "promotions" / f"{loop_id}.jsonl"

    # --- loops ---
    def save_loop(self, genome: LoopGenome) -> None:
        atomic_write_text(self._loop_path(genome.id), _dump_yaml(genome.model_dump(mode="json")))

    def load_loop(self, loop_id: str) -> LoopGenome:
        path = self._loop_path(loop_id)
        if not path.exists():
            raise NotFoundError(f"loop not found: {loop_id}")
        return LoopGenome.model_validate(_yaml.load(path.read_text(encoding="utf-8")))

    def list_loops(self) -> list[LoopGenome]:
        d = self.root / "loops"
        if not d.exists():
            return []
        return [self.load_loop(p.stem) for p in sorted(d.glob("*.yaml"))]

    # --- traces ---
    def append_trace(self, trace: LoopTrace) -> None:
        _append_line(self._traces_path(trace.loop_id), trace.model_dump_json())

    def list_traces(self, loop_id: str, limit: int | None = None) -> list[LoopTrace]:
        path = self._traces_path(loop_id)
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if limit is not None:
            # limit=0 means "none"; `lines[-0:]` would wrongly return everything.
            lines = lines[-limit:] if limit > 0 else []
        return [LoopTrace.model_validate_json(ln) for ln in lines]

    # --- variants ---
    def save_variant(self, variant: LoopVariant) -> None:
        atomic_write_text(
            self._variant_path(variant.loop_id, variant.id),
            _dump_yaml(variant.model_dump(mode="json")),
        )

    def load_variant(self, loop_id: str, variant_id: str) -> LoopVariant:
        path = self._variant_path(loop_id, variant_id)
        if not path.exists():
            raise NotFoundError(f"variant not found: {loop_id}/{variant_id}")
        return LoopVariant.model_validate(_yaml.load(path.read_text(encoding="utf-8")))

    def list_variants(self, loop_id: str) -> list[LoopVariant]:
        d = self.root / "variants" / loop_id
        if not d.exists():
            return []
        return [self.load_variant(loop_id, p.stem) for p in sorted(d.glob("*.yaml"))]

    # --- promotions ---
    def record_promotion(self, record: PromotionRecord) -> None:
        _append_line(self._promotions_path(record.loop_id), record.model_dump_json())

    def list_promotions(self, loop_id: str) -> list[PromotionRecord]:
        path = self._promotions_path(loop_id)
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [PromotionRecord.model_validate_json(ln) for ln in lines]
