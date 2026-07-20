"""Tests for the whystuck setup wizard (whystuck.setup).

The wizard's logic is pure/file-writing functions taking explicit paths, so the
whole thing is unit-testable against tmp dirs — no interactive prompts, no real
client installs. The thin interactive shell in main() is not unit-tested.
"""

import json
from pathlib import Path

from whystuck.setup import (
    MARKER_BEGIN,
    append_marked_block,
    codex_toml_block,
    configure_claude_code,
    configure_codex,
    configure_cursor,
    cursor_mdc,
    from_source_entry,
    markdown_block,
    merge_mcp_servers,
    run,
    server_entry,
    upsert_codex_toml,
    upsert_json_config,
)


class TestEntries:
    def test_server_entry_default_is_uvx(self):
        assert server_entry() == {"command": "uvx", "args": ["whystuck"]}

    def test_from_source_entry_uses_uv_run_directory(self):
        entry = from_source_entry("/abs/whystuck")
        assert entry == {
            "command": "uv",
            "args": ["run", "--directory", "/abs/whystuck", "whystuck"],
        }


class TestMergeMcpServers:
    def test_adds_whystuck_and_preserves_others(self):
        existing = {"mcpServers": {"other": {"command": "x", "args": []}}}
        merged = merge_mcp_servers(existing, "whystuck", server_entry())
        assert set(merged["mcpServers"]) == {"other", "whystuck"}
        assert merged["mcpServers"]["whystuck"]["command"] == "uvx"

    def test_handles_none_and_empty(self):
        merged = merge_mcp_servers(None, "whystuck", server_entry())
        assert merged["mcpServers"]["whystuck"] == server_entry()


class TestUpsertJsonConfig:
    def test_creates_file_with_parents(self, tmp_path: Path):
        p = tmp_path / "nested" / ".mcp.json"
        assert upsert_json_config(p, "whystuck", server_entry()) is True
        data = json.loads(p.read_text())
        assert data["mcpServers"]["whystuck"]["command"] == "uvx"

    def test_preserves_existing_servers(self, tmp_path: Path):
        p = tmp_path / ".mcp.json"
        p.write_text(json.dumps({"mcpServers": {"other": {"command": "x", "args": []}}}))
        upsert_json_config(p, "whystuck", server_entry())
        data = json.loads(p.read_text())
        assert "other" in data["mcpServers"]
        assert "whystuck" in data["mcpServers"]

    def test_idempotent_second_call_is_noop(self, tmp_path: Path):
        p = tmp_path / ".mcp.json"
        assert upsert_json_config(p, "whystuck", server_entry()) is True
        assert upsert_json_config(p, "whystuck", server_entry()) is False

    def test_updates_when_entry_changes(self, tmp_path: Path):
        p = tmp_path / ".mcp.json"
        upsert_json_config(p, "whystuck", server_entry())
        assert upsert_json_config(p, "whystuck", from_source_entry("/x")) is True
        data = json.loads(p.read_text())
        assert data["mcpServers"]["whystuck"]["command"] == "uv"


class TestAppendMarkedBlock:
    def test_appends_once_and_skips_on_rerun(self, tmp_path: Path):
        p = tmp_path / "CLAUDE.md"
        p.write_text("# My project\n")
        assert append_marked_block(p, markdown_block()) is True
        assert MARKER_BEGIN in p.read_text()
        assert append_marked_block(p, markdown_block()) is False
        assert p.read_text().count(MARKER_BEGIN) == 1
        assert "# My project" in p.read_text()  # original preserved

    def test_creates_file_if_missing(self, tmp_path: Path):
        p = tmp_path / "AGENTS.md"
        assert append_marked_block(p, markdown_block()) is True
        assert p.exists()


class TestCodexToml:
    def test_block_is_valid_toml_shape(self):
        block = codex_toml_block("whystuck", server_entry())
        assert "[mcp_servers.whystuck]" in block
        assert 'command = "uvx"' in block
        assert 'args = ["whystuck"]' in block

    def test_appends_and_preserves_and_skips(self, tmp_path: Path):
        p = tmp_path / "config.toml"
        p.write_text('model = "gpt-5"\n')
        assert upsert_codex_toml(p, "whystuck", server_entry()) is True
        txt = p.read_text()
        assert "[mcp_servers.whystuck]" in txt
        assert 'model = "gpt-5"' in txt  # existing config preserved
        assert upsert_codex_toml(p, "whystuck", server_entry()) is False  # idempotent


class TestCursorMdc:
    def test_mdc_has_frontmatter_and_contract(self):
        mdc = cursor_mdc()
        assert mdc.startswith("---")
        assert "alwaysApply: true" in mdc
        assert "whystuck_diagnose" in mdc


class TestConfigureClients:
    def test_configure_claude_code_writes_mcp_and_rules(self, tmp_path: Path):
        changed = configure_claude_code(
            tmp_path / ".mcp.json", tmp_path / "CLAUDE.md", server_entry()
        )
        assert (tmp_path / ".mcp.json").exists()
        assert (tmp_path / "CLAUDE.md").exists()
        assert len(changed) == 2

    def test_configure_cursor_writes_mcp_and_mdc(self, tmp_path: Path):
        changed = configure_cursor(
            tmp_path / ".cursor" / "mcp.json",
            tmp_path / ".cursor" / "rules" / "whystuck.mdc",
            server_entry(),
        )
        assert (tmp_path / ".cursor" / "mcp.json").exists()
        assert (tmp_path / ".cursor" / "rules" / "whystuck.mdc").exists()
        assert len(changed) == 2

    def test_configure_codex_writes_toml_and_agents(self, tmp_path: Path):
        changed = configure_codex(
            tmp_path / ".codex" / "config.toml", tmp_path / "AGENTS.md", server_entry()
        )
        assert (tmp_path / ".codex" / "config.toml").exists()
        assert (tmp_path / "AGENTS.md").exists()
        assert len(changed) == 2


class TestRun:
    def test_run_all_clients_project_scope(self, tmp_path: Path):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        summary = run(
            {"claude", "cursor", "codex"},
            server_entry(),
            project_root=proj,
            home=home,
            scope="project",
        )
        assert (proj / ".mcp.json").exists()
        assert (proj / ".cursor" / "mcp.json").exists()
        assert (proj / ".cursor" / "rules" / "whystuck.mdc").exists()
        assert (home / ".codex" / "config.toml").exists()  # codex mcp is always global
        assert (proj / "AGENTS.md").exists()
        assert set(summary) == {"claude", "cursor", "codex"}

    def test_run_global_scope_writes_home_configs(self, tmp_path: Path):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        run({"claude", "cursor"}, server_entry(), project_root=proj, home=home, scope="global")
        assert (home / ".claude.json").exists()
        assert (home / ".cursor" / "mcp.json").exists()
        assert not (proj / ".mcp.json").exists()

    def test_run_is_idempotent(self, tmp_path: Path):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        run({"claude", "cursor", "codex"}, server_entry(), project_root=proj, home=home)
        second = run(
            {"claude", "cursor", "codex"}, server_entry(), project_root=proj, home=home
        )
        # nothing changed on the second run
        assert all(files == [] for files in second.values())
