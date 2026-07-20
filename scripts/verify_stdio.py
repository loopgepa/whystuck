"""Prove the console script serves MCP over stdio — the `uvx whystuck` contract.

Spawns the real `whystuck` entry point as a subprocess and speaks the MCP stdio
handshake through FastMCP's client. If this lists the tools, then
`claude mcp add whystuck -- uvx whystuck` works identically.

Run:  uv run python scripts/verify_stdio.py
"""

from __future__ import annotations

import asyncio
import sys

from fastmcp import Client


async def main() -> None:
    client = Client({"mcpServers": {"whystuck": {"command": "uv", "args": ["run", "whystuck"]}}})
    async with client:
        names = sorted(t.name for t in await client.list_tools())
    if not names:
        print("stdio FAILED — 0 tools registered", file=sys.stderr)
        raise SystemExit(1)
    print(f"stdio OK — {len(names)} tools registered:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
