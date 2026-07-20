#!/usr/bin/env bash
# Publish-day only: lock down `main` after the repo is pushed to GitHub.
# Run authenticated as the org owner (vladimirrott), after at least one CI run so
# the "test" status-check context exists.
#
#   bash scripts/setup_branch_protection.sh [owner/repo]
#
# Enforces: PR + 1 approving review, CI "test" must pass (strict/up-to-date),
# admins included, no force-push, no branch deletion.
set -euo pipefail
REPO="${1:-loopgepa/loopgepa}"

gh api -X PUT "repos/${REPO}/branches/main/protection" \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["test"] },
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 1 },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

echo "branch protection applied to ${REPO}@main"
