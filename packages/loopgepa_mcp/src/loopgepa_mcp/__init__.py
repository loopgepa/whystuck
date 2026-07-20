"""LoopGEPA MCP server.

Thin, deterministic stdio tool layer over loopgepa-core. No LLM calls inside the
server (ADR: reflection happens in Claude, not here). See IMPLEMENTATION.md §8.
"""

__version__ = "0.1.0.dev0"
