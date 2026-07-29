#!/usr/bin/env bash
# Full local gate — run before every commit. Mirrors CI.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run mypy src/whystuck
uv run pytest
# The paper's numbers are derived from docs/*.json and the committed cassettes, and have
# drifted from them before. Re-derive rather than trust.
uv run python scripts/verify_paper_claims.py | tail -n 3
echo "dev_check: all green"
