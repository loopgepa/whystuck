"""WhyStuck MCP server entry point (`uvx whystuck` → this `main`).

stdio transport (ADR-007). `main()` calls `mcp.run()` unconditionally — the
console-script entry point imports this module and calls `main()` directly, so a
lone `if __name__ == "__main__"` guard would never fire for `uvx whystuck`.
"""

from __future__ import annotations

from whystuck.core.storage import WhyStuckStore
from whystuck.tools import build_server


def main() -> None:
    build_server(WhyStuckStore.from_env()).run()  # stdio is FastMCP's default transport


if __name__ == "__main__":
    main()
