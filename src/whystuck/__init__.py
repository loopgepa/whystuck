"""WhyStuck — local, zero-telemetry diagnosis and reversible repair of
coding-agent loops, exposed as an MCP server for Claude Code / Codex / Cursor.

Public distribution is a single package so that `uvx whystuck` just works.
Vendor-neutral core logic lives in `whystuck.core`; the MCP/tool layer wraps it.
See IMPLEMENTATION.md and DECISIONS.md.
"""

__version__ = "0.1.0.dev0"
