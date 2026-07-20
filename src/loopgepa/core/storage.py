"""LoopGEPA local file store (ADR-004).

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
from collections.abc import Mapping
from pathlib import Path

from ruamel.yaml import YAML

from loopgepa.core.models import (
    LoopGenome,
    LoopTrace,
    LoopVariant,
    PromotionRecord,
)

_yaml = YAML(typ="safe")
_yaml.default_flow_style = False


def resolve_store_root(env: Mapping[str, str]) -> Path:
    """Resolve the store root: LOOPGEPA_HOME > $XDG_DATA_HOME/loopgepa >
    ~/.local/share/loopgepa. Never the current working directory (ADR-004.2)."""
    if home := env.get("LOOPGEPA_HOME"):
        return Path(home)
    if xdg := env.get("XDG_DATA_HOME"):
        return Path(xdg) / "loopgepa"
    return Path(env.get("HOME", "~")) / ".local" / "share" / "loopgepa"


def _dump_yaml(data: object) -> str:
    buf = io.StringIO()
    _yaml.dump(data, buf)
    return buf.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + os.replace so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


class LoopGEPAStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LoopGEPAStore:
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
        _atomic_write(self._loop_path(genome.id), _dump_yaml(genome.model_dump(mode="json")))

    def load_loop(self, loop_id: str) -> LoopGenome:
        path = self._loop_path(loop_id)
        if not path.exists():
            raise KeyError(f"loop not found: {loop_id}")
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
            lines = lines[-limit:]
        return [LoopTrace.model_validate_json(ln) for ln in lines]

    # --- variants ---
    def save_variant(self, variant: LoopVariant) -> None:
        _atomic_write(
            self._variant_path(variant.loop_id, variant.id),
            _dump_yaml(variant.model_dump(mode="json")),
        )

    def load_variant(self, loop_id: str, variant_id: str) -> LoopVariant:
        path = self._variant_path(loop_id, variant_id)
        if not path.exists():
            raise KeyError(f"variant not found: {loop_id}/{variant_id}")
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
