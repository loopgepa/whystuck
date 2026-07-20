# Comparison

Every established LLM-observability / eval tool is **cloud SaaS (or a self-hosted
server) aimed at production LLM applications**. LoopGEPA is local, zero-telemetry,
and scoped to a developer's own coding-agent loops. Sources and full profiles:
[Competitive & launch notes](COMPETITIVE-AND-LAUNCH.md).

| Tool | Hosting | Scope | Proposes a fix? | Local / no-telemetry |
|---|---|---|---|---|
| **LoopGEPA** | **Local (stdio MCP)** | **Coding-agent loops** | **Yes — reversible, you approve** | **Yes** |
| Latitude | Cloud SaaS (+ self-host) | Production agents | Auto-PR (no approval gate) | No |
| LangSmith | Cloud SaaS | Production LLM apps | No | No |
| Braintrust | Cloud SaaS | Production LLM apps | No | No |
| Langfuse | Cloud + self-host | Production LLM apps | No | Self-host only |
| Arize Phoenix | Self-host server | Production LLM apps | No | Server required |

**What nobody else pairs:** *local + zero-telemetry + coding-agent-session scope +
reversible, approval-gated repair.*

Closest analogs, and why they differ:

- **Latitude** clusters failures and dispatches an agent to open a PR — but it's
  cloud/production-scoped and auto-applies without a human approval gate.
- **Perseval** is genuinely local, but generates eval *definitions* only — it does
  not propose an actual fix.

Adjacent, not competing:

- **`loop-engineering`** (Cobus Greyling) helps you *build* loops (scaffold,
  audit, cost). LoopGEPA diagnoses *why a loop fails* and repairs it.
- **`gepa-ai/gepa`** is the optimizer LoopGEPA's v0.2 will *wrap*, not compete
  with — see [v0.2 design](v0.2-eval-benchmark.md).
