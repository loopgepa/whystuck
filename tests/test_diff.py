"""Tests for loopgepa.core.diff — human-readable gene diffs."""

from loopgepa.core.diff import render_gene_diff


class TestRenderGeneDiff:
    def test_shows_added_and_removed_lines(self):
        diff = render_gene_diff("rubric", "check it", "check it thoroughly")
        assert "-check it" in diff
        assert "+check it thoroughly" in diff

    def test_includes_gene_name_header(self):
        diff = render_gene_diff("rubric", "a", "b")
        assert "rubric" in diff

    def test_no_change_returns_empty(self):
        assert render_gene_diff("rubric", "same", "same") == ""
