#!/usr/bin/env bash
# Full local gate — run before every commit. Mirrors CI.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run mypy src/loopgepa
uv run pytest
echo "dev_check: all green"
