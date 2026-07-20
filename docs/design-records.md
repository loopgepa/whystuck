# Design records (ADRs)

The full architecture decision record is [`DECISIONS.md`](https://github.com/loopgepa/whystuck/blob/main/DECISIONS.md)
(single source of truth). Index:

- **ADR-001** — Name: `whystuck` (renamed from LoopGEPA; name = command = function)
- **ADR-002** — WhyStuck owns the loop runner, so every gene is a real knob
- **ADR-003** — v0.1 scope cut: no Pareto/eval (no fuel without measurement)
- **ADR-004** — Trust model: allowlist storage, not denylist redaction
- **ADR-005** — Confidence is deterministic, not vibes
- **ADR-006** — One PyPI package; `uvx whystuck` is the whole setup
- **ADR-007** — FastMCP v3, stdio only
- **ADR-008** — The MCP server is the product; the plugin is thin
- **ADR-009** — OSS-quality standards adopted from SysKnife (staged to launch)
- **ADR-010** — Positioning: name the pain; local-first; anti-hype
- **ADR-011** — v0.2 evaluation wraps `gepa-ai/gepa` (don't reinvent Pareto)
- **ADR-012** — Position as the diagnosis/repair layer for *loop engineering*
- **ADR-013** — Design principles distilled from competitor-failure research
