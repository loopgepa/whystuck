#!/usr/bin/env bash
# Version-drift guard: the pyproject base version must match the plugin manifest.
# (pyproject may carry a .devN suffix; we compare the base X.Y.Z.)
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(grep -oP '(?<=^version = ")[^"]+' pyproject.toml | head -1)
py_base=${py%%.dev*}
py_base=${py_base%%[-+]*}
pl=$(grep -oP '(?<="version": ")[^"]+' plugin/.claude-plugin/plugin.json | head -1)

echo "check_versions: pyproject=$py (base $py_base)  plugin.json=$pl"
if [ "$py_base" != "$pl" ]; then
  echo "FAIL: version drift between pyproject.toml and plugin.json" >&2
  exit 1
fi
echo "check_versions: OK"
