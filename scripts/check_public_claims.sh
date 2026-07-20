#!/usr/bin/env bash
# Verify README's test-count claim is not an overstatement (ADR-009).
# The README may claim "N tests"; the actual collected count must be >= N.
set -euo pipefail
cd "$(dirname "$0")/.."

claimed=$(grep -oiP '\b\d+(?=\s+tests\b)' README.md | head -1 || true)
if [ -z "${claimed:-}" ]; then
  echo "check_public_claims: no test-count claim in README (ok)"
  exit 0
fi

# Sum the per-file counts from `pytest --collect-only -q` (e.g. "tests/x.py: 6").
# stdout only — folding in stderr would let a warning ending in ": <n>" inflate
# the tally and hide a real shortfall. A collect error aborts loudly via set -e.
collected=$(uv run pytest --collect-only -q 2>/dev/null)
actual=$(printf '%s\n' "$collected" | grep -oE ': [0-9]+$' | grep -oE '[0-9]+' \
  | awk '{s+=$1} END{print s+0}')
actual=${actual:-0}
echo "check_public_claims: README claims ${claimed} tests; actual ${actual}"
if [ "${actual}" -lt "${claimed}" ]; then
  echo "FAIL: README claims more tests (${claimed}) than exist (${actual})" >&2
  exit 1
fi
echo "check_public_claims: OK"
