# Roadmap

## v0.1 — Diagnose + Suggest + Promote (local MVP)

The honest, defensible core. No evaluation, no Pareto (ADR-003).

- Loop genome + trace models; local XDG file store.
- Runner that executes declared loops and emits clean traces.
- Deterministic failure-mode aggregation + confidence (ADR-005).
- Reflective mutation (≤3 genes) with evidence trace IDs.
- Diff preview → explicit approval → reversible promote + rollback record.
- FastMCP stdio server; `uvx loopgepa`; one bugfix-loop example.

## v0.2 — Evaluate (earn the GEPA claim)

- Live variant evaluation on task sets.
- Wrap `gepa-ai/gepa` for reflective optimization + Pareto selection over
  multi-objective metrics (success, cost, latency, diff size, safety).
- Only now may output be called "optimized."

## v0.3+ — Reach & polish

- Optional Claude Code plugin marketplace listing + slash commands.
- mkdocs-material docs site; benchmark: prompt-only GEPA vs loop-level GEPA.
- Possible `.mcpb` desktop bundle if Claude Desktop demand appears.

Non-goals (indefinitely): cloud backend, telemetry, silent promotion,
arbitrary loop-topology mutation.
