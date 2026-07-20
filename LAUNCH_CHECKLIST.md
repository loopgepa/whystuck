# LoopGEPA — Pre-Public Launch Gate

**Policy:** the repo stays 100% local until every box in "MVP" and "Launch
hardening" is checked. Goal: one strong first impression for a viral OSS launch,
not a public pre-alpha. Rationale and sources: `DECISIONS.md` (ADR-009, ADR-010)
and the SysKnife reference repo (`/home/user/Desktop/lacs`).

## Gate 1 — MVP (Definition of Done, IMPLEMENTATION.md §12)

- [ ] `models.py` + tests (M1)
- [ ] `storage.py` (XDG root, JSONL append, schema_version) + tests (M2)
- [ ] `mutation.py` + `diff.py` + tests (M3)
- [ ] `runner.py` (minimal) emits clean traces + tests (M4)
- [ ] FastMCP server wraps core over stdio + smoke tests (M5)
- [ ] `uvx loopgepa` cold-starts cleanly from a built wheel in a fresh env
- [ ] `claude mcp add loopgepa -- uvx loopgepa` verified end-to-end
- [ ] promote dry-run + real promote + rollback record + tests (M7)
- [ ] `examples/bugfix-loop` full walkthrough works (M8)
- [ ] optional thin plugin: `/loopgepa:explain-loop-failure` slash command (M6)

## Gate 2 — Launch hardening (copied from SysKnife, staged for launch)

**CI / security**
- [ ] SHA-pin every GitHub Action with `# vX` comment
- [ ] commit `uv.lock`; CI uses `uv sync --frozen`
- [ ] split fast gate (lint/type/test) vs `workflow_dispatch` LLM-behavior tests
- [ ] `codeql.yml` (python), `scorecard.yml`, `secret-scan.yml` (TruffleHog
      `--results=verified`), dependency-review on PRs
- [ ] `scripts/check_public_claims.sh` verifies README claims vs reality

**Docs**
- [ ] `mkdocs-material` site + `docs.yml` Pages deploy (paths-filtered)
- [ ] docs nav: Getting Started / How It Works / Safety / Design Records (ADRs)
- [ ] ADRs mirrored from `DECISIONS.md` into `docs/adr/`

**Release**
- [ ] `release.yml`: tag-triggered, `preflight` requires PyPI token, idempotent
      (check-then-skip if version exists), SBOM + provenance attestation
- [ ] version-drift check (pyproject vs plugin.json vs CHANGELOG)
- [ ] `release-rehearsal.yml` (dispatch, builds + packages without publishing)

**Virality assets**
- [ ] hero GIF (≤30s, ≤5MB) from a checked-in VHS `.tape`: agent loops on a
      failing command → LoopGEPA diagnoses → proposes reversible fix → approve
- [ ] README badge row wired (CI, license, PyPI version, stars)
- [ ] concrete test-count line in README (verified by check_public_claims)

## Gate 3 — Publish day (in order)

- [ ] register PyPI name `loopgepa`; `uv publish`
- [ ] switch `gh` to **vladimirrott**; create `loopgepa/loopgepa` repo; push
- [ ] enable branch protection on `main` (PR + 1 review + CI required)
- [ ] enable GitHub Pages, Discussions, Dependabot, code scanning
- [ ] submit to MCP.Directory, Anthropic MCP directory, Claude Code plugin
      marketplace
- [ ] launch post naming the whitespace vs Latitude's "GEPA algorithm" (local +
      zero-telemetry + coding-agent-session scope)
