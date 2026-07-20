"""`whystuck setup` — a one-command wizard that wires WhyStuck into your MCP
clients (Claude Code, Codex, Cursor) and teaches the agent when to reach for it.

Design: WhyStuck already requires `uv` (the server runs as `uvx whystuck`), so the
wizard stays in that world — no Node, no extra prerequisite. All the real work is
pure/file-writing functions taking explicit paths, so it is fully unit-testable;
`main()` is a thin interactive shell on top.

The best idea borrowed from SysKnife's installer: don't just write the MCP config,
also inject a short *usage contract* in each client's native rule format (CLAUDE.md,
Cursor `.mdc`, AGENTS.md) so the agent knows to diagnose a stuck loop before
thrashing further — correct usage becomes the path of least resistance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

MARKER_BEGIN = "<!-- whystuck:begin -->"
MARKER_END = "<!-- whystuck:end -->"

# The usage contract, injected verbatim into every client's rule file. Plain prose
# (no headings) so it embeds cleanly in Markdown and Cursor `.mdc` alike.
CONTRACT = (
    "When a loop keeps failing — repeated edits, retries that don't help, or the "
    "same error after several attempts — use the WhyStuck MCP tools to diagnose "
    "*why* before changing more code. Call `whystuck_diagnose` for the dominant "
    "failure mode and its confidence, then `whystuck_propose_fix` for a reversible "
    "≤3-gene patch, and apply it with `whystuck_promote` (dry-run first — you "
    "approve every write)."
)


# --- config values ------------------------------------------------------------


def server_entry(command: str = "uvx", args: list[str] | None = None) -> dict[str, Any]:
    """The MCP server entry for a published install: `uvx whystuck` over stdio."""
    return {"command": command, "args": args if args is not None else ["whystuck"]}


def from_source_entry(path: str) -> dict[str, Any]:
    """The MCP server entry for a pre-PyPI local clone: `uv run --directory <path>`."""
    return {"command": "uv", "args": ["run", "--directory", path, "whystuck"]}


def merge_mcp_servers(
    existing: dict[str, Any] | None, name: str, entry: dict[str, Any]
) -> dict[str, Any]:
    """Add/replace one server under `mcpServers`, preserving every other key —
    critical when merging into a shared file like ~/.claude.json."""
    base = dict(existing) if isinstance(existing, dict) else {}
    servers = dict(base.get("mcpServers") or {})
    servers[name] = entry
    base["mcpServers"] = servers
    return base


def markdown_block() -> str:
    """The marker-delimited contract block for CLAUDE.md / AGENTS.md."""
    return f"{MARKER_BEGIN}\n## WhyStuck — diagnose stuck loops\n\n{CONTRACT}\n{MARKER_END}"


def cursor_mdc() -> str:
    """A Cursor project rule (`.mdc`) carrying the same contract, always applied."""
    return (
        "---\n"
        "description: Diagnose repeatedly failing agent loops with WhyStuck\n"
        "alwaysApply: true\n"
        "---\n\n"
        f"{CONTRACT}\n"
    )


def codex_toml_block(name: str, entry: dict[str, Any]) -> str:
    """A `[mcp_servers.<name>]` block for ~/.codex/config.toml (hand-rendered TOML;
    json.dumps yields valid TOML string/array literals)."""
    args_toml = "[" + ", ".join(json.dumps(a) for a in entry["args"]) + "]"
    return (
        "# --- whystuck (added by whystuck setup) ---\n"
        f"[mcp_servers.{name}]\n"
        f"command = {json.dumps(entry['command'])}\n"
        f"args = {args_toml}\n"
    )


# --- file writers (idempotent) ------------------------------------------------


def upsert_json_config(path: Path, name: str, entry: dict[str, Any]) -> bool:
    """Merge one MCP server into a JSON config, preserving the rest. Returns True
    iff the file changed (unchanged entry → no write)."""
    existing: Any = None
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
    servers = existing.get("mcpServers") if isinstance(existing, dict) else None
    if isinstance(servers, dict) and servers.get(name) == entry:
        return False
    merged = merge_mcp_servers(existing if isinstance(existing, dict) else None, name, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return True


def append_marked_block(path: Path, block: str, begin: str = MARKER_BEGIN) -> bool:
    """Append a marker-delimited block once. Returns False (no-op) if the marker is
    already present. Creates the file (and parents) if missing."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if begin in existing:
        return False
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix:
        prefix += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix + block + "\n", encoding="utf-8")
    return True


def upsert_codex_toml(path: Path, name: str, entry: dict[str, Any]) -> bool:
    """Append the `[mcp_servers.<name>]` block once, preserving existing config.
    Returns False if the block is already present."""
    marker = f"[mcp_servers.{name}]"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return False
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix:
        prefix += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prefix + codex_toml_block(name, entry), encoding="utf-8")
    return True


def write_if_changed(path: Path, content: str) -> bool:
    """Write `content` only if it differs from what's on disk. Returns True iff written."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


# --- per-client configuration -------------------------------------------------


def configure_claude_code(mcp_path: Path, rules_path: Path, entry: dict[str, Any]) -> list[str]:
    changed = []
    if upsert_json_config(mcp_path, "whystuck", entry):
        changed.append(str(mcp_path))
    if append_marked_block(rules_path, markdown_block()):
        changed.append(str(rules_path))
    return changed


def configure_cursor(mcp_path: Path, mdc_path: Path, entry: dict[str, Any]) -> list[str]:
    changed = []
    if upsert_json_config(mcp_path, "whystuck", entry):
        changed.append(str(mcp_path))
    if write_if_changed(mdc_path, cursor_mdc()):
        changed.append(str(mdc_path))
    return changed


def configure_codex(toml_path: Path, agents_path: Path, entry: dict[str, Any]) -> list[str]:
    changed = []
    if upsert_codex_toml(toml_path, "whystuck", entry):
        changed.append(str(toml_path))
    if append_marked_block(agents_path, markdown_block()):
        changed.append(str(agents_path))
    return changed


def run(
    clients: set[str],
    entry: dict[str, Any],
    *,
    project_root: Path,
    home: Path,
    scope: str = "project",
) -> dict[str, list[str]]:
    """Configure the selected clients. Returns {client: [files changed]} — an empty
    list means that client was already configured. `scope` is "project" (write into
    project_root) or "global" (write user-level config). Codex's MCP config is
    always global (~/.codex/config.toml); its AGENTS.md rule stays project-local."""
    summary: dict[str, list[str]] = {}
    if "claude" in clients:
        if scope == "global":
            mcp, rules = home / ".claude.json", home / ".claude" / "CLAUDE.md"
        else:
            mcp, rules = project_root / ".mcp.json", project_root / "CLAUDE.md"
        summary["claude"] = configure_claude_code(mcp, rules, entry)
    if "cursor" in clients:
        if scope == "global":
            mcp, mdc = home / ".cursor" / "mcp.json", home / ".cursor" / "rules" / "whystuck.mdc"
        else:
            mcp = project_root / ".cursor" / "mcp.json"
            mdc = project_root / ".cursor" / "rules" / "whystuck.mdc"
        summary["cursor"] = configure_cursor(mcp, mdc, entry)
    if "codex" in clients:
        summary["codex"] = configure_codex(
            home / ".codex" / "config.toml", project_root / "AGENTS.md", entry
        )
    return summary


# --- interactive shell --------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _ok(t: str) -> str:
    return _c("✓ ", "32") + t


def _warn(t: str) -> str:
    return _c("⚠ ", "33") + t


def _arrow(t: str) -> str:
    return _c("→ ", "36") + t


def _prompt_clients() -> set[str]:
    print("Which clients should I configure?")
    print("  1) Claude Code\n  2) Cursor\n  3) Codex CLI\n  4) All three  [default]")
    try:
        choice = input("> ").strip() or "4"
    except EOFError:
        return {"claude", "cursor", "codex"}
    return {
        "1": {"claude"},
        "2": {"cursor"},
        "3": {"codex"},
        "4": {"claude", "cursor", "codex"},
    }.get(choice, {"claude", "cursor", "codex"})


def _print_summary(summary: dict[str, list[str]]) -> None:
    any_change = False
    for client, files in sorted(summary.items()):
        if files:
            any_change = True
            print(_ok(f"{client}: configured"))
            for f in files:
                print(f"    wrote {f}")
        else:
            print(_arrow(f"{client}: already configured (no changes)"))
    print()
    print(_ok("Setup complete.") if any_change else _ok("Already set up — nothing to change."))
    print(
        _arrow(
            "Reload MCP servers (Claude Code: restart or /reload-plugins · "
            "Cursor: Settings → MCP → Refresh · Codex: picked up next run)."
        )
    )
    print(_arrow('Try it: ask your agent "why does my loop keep failing?"'))
    print(
        _arrow(
            "If `uvx whystuck` fails (not yet on PyPI), re-run with "
            "--from-source /path/to/whystuck."
        )
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="whystuck setup", description="Wire WhyStuck into your MCP clients."
    )
    parser.add_argument("--all", action="store_true", help="Claude Code + Cursor + Codex")
    parser.add_argument("--claude", action="store_true", help="configure Claude Code")
    parser.add_argument("--cursor", action="store_true", help="configure Cursor")
    parser.add_argument("--codex", action="store_true", help="configure Codex CLI")
    parser.add_argument(
        "--global",
        dest="global_scope",
        action="store_true",
        help="write user-level config (default: this project)",
    )
    parser.add_argument("--project-dir", default=".", help="project root (default: cwd)")
    parser.add_argument(
        "--from-source",
        metavar="PATH",
        help="run from a local clone instead of uvx (pre-PyPI)",
    )
    parser.add_argument(
        "--yes",
        "--no-prompts",
        dest="yes",
        action="store_true",
        help="non-interactive; default to all clients",
    )
    ns = parser.parse_args(argv)

    clients: set[str] = set()
    if ns.all:
        clients = {"claude", "cursor", "codex"}
    else:
        if ns.claude:
            clients.add("claude")
        if ns.cursor:
            clients.add("cursor")
        if ns.codex:
            clients.add("codex")
    if not clients:
        clients = {"claude", "cursor", "codex"} if ns.yes else _prompt_clients()
    if not clients:
        print(_warn("No clients selected — nothing to do."))
        return 0

    entry = (
        from_source_entry(str(Path(ns.from_source).resolve()))
        if ns.from_source
        else server_entry()
    )
    scope = "global" if ns.global_scope else "project"
    project_root = Path(ns.project_dir).resolve()

    print(_c("WhyStuck setup", "1"))
    print(
        _arrow(
            f"clients: {', '.join(sorted(clients))}  ·  scope: {scope}  ·  "
            f"command: {entry['command']} {' '.join(entry['args'])}"
        )
    )
    summary = run(clients, entry, project_root=project_root, home=Path.home(), scope=scope)
    _print_summary(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
