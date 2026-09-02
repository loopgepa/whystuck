#!/usr/bin/env bash
# Build all three double-blind submission packages against a published mirror URL.
#
# Everything up to this point is done and verified. This exists so the last step is one command
# with one argument, rather than three invocations with a URL to keep retyping.
#
#   ./scripts/anon_submission.sh https://anonymous.4open.science/r/<id>
#
# Each build strips the author block and the repository URL, then scans every byte of the staged
# tree for six identifying strings and refuses to write a tarball if one survives.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ $# -ne 1 ]; then
  echo "usage: $0 <anonymous mirror URL>" >&2
  echo >&2
  echo "Build the mirror first, then publish it and pass its URL here:" >&2
  echo "  python scripts/make_anon_mirror.py --dest dist/anon-mirror" >&2
  exit 2
fi
URL="$1"

case "$URL" in
  http://*|https://*) ;;
  *) echo "refusing: '$URL' is not a URL. The abstract claims a release; a claim pointing at nothing is worse than a named one." >&2; exit 2 ;;
esac
for token in whystuck loopgepa vladimirrott Vladimir Rotariu; do
  if printf '%s' "$URL" | grep -qi -- "$token"; then
    echo "refusing: the mirror URL contains '$token', which de-anonymises the submission it is meant to support." >&2
    exit 2
  fi
done

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
for paper in p1-conditional-lever p2-repair-not-detection p3-silent-no-op; do
  echo "=== $paper"
  "$PY" scripts/make_arxiv_package.py --paper "$paper" --anonymous --anon-url "$URL"
done
