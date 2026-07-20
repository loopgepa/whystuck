# LoopGEPA — Pre-Public Launch Gate

**Policy:** the repo stays 100% local until every box in "MVP" and "Launch
hardening" is checked. Goal: one strong first impression for a viral OSS launch,
not a public pre-alpha. Rationale and sources: `DECISIONS.md` (ADR-009, ADR-010)
and the SysKnife reference repo (`/home/user/Desktop/lacs`).

## Gate 1 — MVP (Definition of Done, IMPLEMENTATION.md §12) — ✅ COMPLETE

- [x] `models.py` + tests (M1)
- [x] `storage.py` (XDG root, JSONL append, schema_version) + tests (M2)
- [x] `mutation.py` + `diff.py` + `safety.py` + tests (M3)
- [x] `diagnosis.py` deterministic failure-mode clustering + confidence (M4a)
- [x] `runner.py` command+verifier ingest emits clean traces + tests (M4)
- [x] FastMCP server wraps core over stdio + in-memory Client smoke tests (M5)
- [x] `uvx loopgepa` / console script serves MCP over stdio (scripts/verify_stdio.py)
- [x] promote dry-run + real promote + rollback record + tests (M7)
- [x] `examples/bugfix-loop` runnable offline demo works (M8)
- [x] thin plugin: explain-loop-failure / optimize-loop / design-loop skills (M6)

_80 tests, ruff + mypy strict, docs build --strict, demo + stdio verified green._

## Gate 2 — Launch hardening (copied from SysKnife, staged for launch)

**CI / security**
- [x] commit `uv.lock`; CI uses `uv sync --frozen`
- [x] `codeql.yml` (python), `scorecard.yml`, `secret-scan.yml` (TruffleHog
      `--results=verified`)
- [x] `scripts/check_public_claims.sh` verifies README test count vs reality
- [ ] SHA-pin every GitHub Action with `# vX` comment (NOTE left in each workflow)
- [ ] split fast gate vs `workflow_dispatch` LLM-behavior tests (add when LLM
      tests exist)
- [ ] dependency-review on PRs

**Docs**
- [x] `mkdocs-material` site (`mkdocs.yml`) builds `--strict`; `docs.yml` Pages deploy
- [x] docs nav: Home / Getting started / Trust & safety / Competitive & launch
- [ ] mirror ADRs from `DECISIONS.md` into `docs/adr/` (optional)

**Release**
- [ ] `release.yml`: tag-triggered, `preflight` requires PyPI token, idempotent
      (check-then-skip if version exists), SBOM + provenance attestation
- [ ] version-drift check (pyproject vs plugin.json vs CHANGELOG)
- [ ] `release-rehearsal.yml` (dispatch, builds + packages without publishing)

**Virality assets**
- [x] README badge row wired (CI, license, PyPI version)
- [x] concrete test-count line in README (verified by check_public_claims)
- [ ] hero GIF (≤30s, ≤5MB) from a checked-in VHS `.tape`: diagnose → propose →
      approve → promote (base it on `examples/bugfix-loop/demo.py`)

## Gate 3 — Publish day (in order)

- [ ] register PyPI name `loopgepa`; `uv publish`
- [ ] switch `gh` to **vladimirrott**; create `loopgepa/loopgepa` repo; push
- [ ] enable branch protection on `main` (PR + 1 review + CI required)
- [ ] enable GitHub Pages, Discussions, Dependabot, code scanning
- [ ] submit to MCP.Directory, Anthropic MCP directory, Claude Code plugin
      marketplace
- [ ] launch post naming the whitespace vs Latitude's "GEPA algorithm" (local +
      zero-telemetry + coding-agent-session scope)
