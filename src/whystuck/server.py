"""WhyStuck MCP server entry point (`uvx whystuck` → this `main`).

stdio transport (ADR-007). MCP clients spawn `whystuck` with no arguments, so the
bare invocation runs the server. `whystuck setup ...` dispatches to the setup
wizard instead (a subcommand keeps everything in one package/entry point).
"""

from __future__ import annotations

import sys

from whystuck.core.storage import WhyStuckStore
from whystuck.tools import build_server


def main() -> None:
    # `whystuck setup [...]` → wizard; anything else (incl. no args, how MCP
    # clients launch it) → the stdio server.
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from whystuck.setup import main as setup_main

        raise SystemExit(setup_main(sys.argv[2:]))
    build_server(WhyStuckStore.from_env()).run()  # stdio is FastMCP's default transport


if __name__ == "__main__":
    main()
