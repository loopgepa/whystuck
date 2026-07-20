"""Tiny shared helpers with no WhyStuck-specific dependencies.

Centralized so the timestamp format and the empty-safe mean live in exactly one
place (they were duplicated across tools, the optimizer, and the eval workers).
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string — the timestamp stored on records."""
    return datetime.now(UTC).isoformat()


def mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence (never divides by zero)."""
    return sum(values) / len(values) if values else 0.0
