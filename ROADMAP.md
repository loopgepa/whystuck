# Roadmap

## v0.1 — Diagnose + Suggest + Promote (local MVP)

The honest, defensible core. No evaluation, no Pareto (ADR-003).

- Loop genome + trace models; local XDG file store.
- Runner that executes declared loops and emits clean traces.
- Deterministic failure-mode aggregation + confidence (ADR-005).
- Reflective mutation (≤3 genes) with evidence trace IDs.
- Diff preview → explicit approval → reversible promote + rollback record.
- FastMCP stdio server; `uvx whystuck`; one bugfix-loop example.
- PostToolUse thrash hook: auto-detects repeated edits and captures session traces.

## v0.2 — Evaluate (earn the GEPA claim) — [design](docs/v0.2-eval-benchmark.md)

Wrap the official `gepa-ai/gepa` optimizer via a `LoopGenomeAdapter`; GEPA owns
the Pareto frontier, WhyStuck owns the loop-genome mapping (ADR-011).

- ✅ `eval/objectives.py` — multi-objective vector + success-dominant scalarizer
- ✅ `eval/benchmark.py` — task set from recorded traces (replay) or a `tasks/` dir
- ✅ `eval/adapter.py` — `LoopGenomeAdapter` (genome↔candidate, offline-testable)
- ✅ `eval/optimize.py` — `gepa.optimize` wrapper → winner variant + frontier
- ✅ MCP tools `optimize_loop` + `get_frontier`, gated on the `eval` extra
- ✅ `eval/llm_worker.py` — LLM-backed worker so text genes move the score
  (`make_run_task(complete, score)`; litellm-backed default)
- 🟡 the output→apply→verify wiring so LLM optimization runs against real repos
- 📋 Only call output "optimized" after evaluation; report cost/latency/variance
- 📋 The A/B/C/D writeup: prompt-only GEPA vs loop-level GEPA (IMPLEMENTATION §16)

## v0.3+ — Reach & polish

- Optional Claude Code plugin marketplace listing + slash commands.
- mkdocs-material docs site; benchmark: prompt-only GEPA vs loop-level GEPA.
- Possible `.mcpb` desktop bundle if Claude Desktop demand appears.

Non-goals (indefinitely): cloud backend, telemetry, silent promotion,
arbitrary loop-topology mutation.
