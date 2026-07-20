"""Prove the console script serves MCP over stdio — the `uvx loopgepa` contract.

Spawns the real `loopgepa` entry point as a subprocess and speaks the MCP stdio
handshake through FastMCP's client. If this lists the tools, then
`claude mcp add loopgepa -- uvx loopgepa` works identically.

Run:  uv run python scripts/verify_stdio.py
"""

from __future__ import annotations

import asyncio

from fastmcp import Client


async def main() -> None:
    client = Client({"mcpServers": {"loopgepa": {"command": "uv", "args": ["run", "loopgepa"]}}})
    async with client:
        names = sorted(t.name for t in await client.list_tools())
    print(f"stdio OK — {len(names)} tools registered:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(main())
