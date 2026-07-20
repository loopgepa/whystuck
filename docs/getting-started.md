# Getting started

## Requirements

[`uv`](https://astral.sh/uv). If `uvx` isn't found:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

=== "Claude Code"

    ```bash
    claude mcp add whystuck -- uvx whystuck
    ```

=== "Codex CLI"

    ```bash
    codex mcp add whystuck -- uvx whystuck
    ```

=== "Cursor / any MCP client"

    ```json
    { "mcpServers": { "whystuck": { "command": "uvx", "args": ["whystuck"] } } }
    ```

No `pip install`, no virtualenv — `uvx` runs it from PyPI in a cached ephemeral
environment.

### Before the PyPI release — run from a local clone

`uvx whystuck` needs the package on PyPI. Until it's published, register the
server from a git clone instead:

```bash
git clone https://github.com/loopgepa/whystuck && cd whystuck
claude mcp add whystuck -- uv run --directory "$PWD" whystuck   # Claude Code
codex  mcp add whystuck -- uv run --directory "$PWD" whystuck   # Codex CLI
```

For Cursor / any MCP client, the equivalent `mcp.json` entry is:

```json
{ "mcpServers": { "whystuck": { "command": "uv", "args": ["run", "--directory", "/abs/path/to/whystuck", "whystuck"] } } }
```

Switch to the `uvx whystuck` commands above once the package is on PyPI.

## Try it (offline, no API key)

```bash
uv run python examples/bugfix-loop/demo.py
```

You'll see: record failing traces → diagnose the dominant failure mode with a
confidence level → propose a reversible ≤3-gene patch → dry-run promotion (diff
only, nothing written).

## Inside your agent

```text
/whystuck:explain-loop-failure bugfix-loop     # diagnose only
/whystuck:optimize-loop bugfix-loop            # propose → dry-run → approve → promote
/whystuck:design-loop                          # declare a new loop
```
