"""Tests for whystuck.eval.bcb_exec._scrub_volatile — the determinism scrub.

BigCodeBench's own tests use ``unittest.mock``; a failing assertion embeds a mock's
repr (``<MagicMock ... id='140...'>``) or a bare object address (``at 0x7f...``),
whose numeric part is a process-local, run-to-run-varying identity. Our in-loop
feedback surfaces that failure text, so an unscrubbed id rides into the next prompt
and shifts the content-addressed cassette key — breaking bit-for-bit replay. The
scrub normalizes those volatile tokens to fixed placeholders so the feedback (hence
the prompt, hence the key) is deterministic. Pure function; no Docker.
"""

from __future__ import annotations

from whystuck.eval.bcb_exec import _scrub_volatile


def test_mock_object_id_is_normalized():
    # The exact string that caused the replay CassetteMiss.
    s = "FAIL test_valid_files: -  <MagicMock name='Popen().returncode' id='126686832446496'>]"
    out = _scrub_volatile(s)
    assert "126686832446496" not in out
    assert "MagicMock" in out and "Popen().returncode" in out  # meaning preserved


def test_two_runs_with_different_ids_scrub_to_the_same_text():
    a = "<MagicMock id='140234567890123'> at 0x7f3c1a2b3c4d"
    b = "<MagicMock id='140987654321000'> at 0x7f0099887766"
    assert _scrub_volatile(a) == _scrub_volatile(b)


def test_bare_hex_address_is_normalized():
    s = "<socket.socket object at 0x7fabc0123def>"
    out = _scrub_volatile(s)
    assert "0x7fabc0123def" not in out
    assert "socket.socket object at" in out


def test_temp_paths_are_normalized():
    a = "FileNotFoundError: /tmp/tmpa1b2c3d4/data.csv"
    b = "FileNotFoundError: /tmp/tmp99887766/data.csv"
    assert _scrub_volatile(a) == _scrub_volatile(b)
    assert "data.csv" in _scrub_volatile(a)  # the stable part survives


def test_idempotent_and_leaves_ordinary_text_untouched():
    s = "AssertionError: 6 != 7 : the sum was wrong"
    assert _scrub_volatile(s) == s
    assert _scrub_volatile(_scrub_volatile(s)) == _scrub_volatile(s)


def test_does_not_touch_short_hex_like_numbers():
    # A genuine small integer difference must survive (only long/hex ids are volatile).
    s = "AssertionError: Lists differ: [0, 1] != [0, 2]"
    assert _scrub_volatile(s) == s
