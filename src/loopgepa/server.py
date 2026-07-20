"""LoopGEPA MCP server entry point (`uvx loopgepa` → this `main`).

stdio transport (ADR-007). `main()` calls `mcp.run()` unconditionally — the
console-script entry point imports this module and calls `main()` directly, so a
lone `if __name__ == "__main__"` guard would never fire for `uvx loopgepa`.
"""

from __future__ import annotations

from loopgepa.core.storage import LoopGEPAStore
from loopgepa.tools import build_server


def main() -> None:
    build_server(LoopGEPAStore.from_env()).run()  # stdio is FastMCP's default transport


if __name__ == "__main__":
    main()
