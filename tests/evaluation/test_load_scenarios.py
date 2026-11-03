"""Tests for the Stage 4 load-test scenarios - small-scale, but against the
real components (SQLite, apply_decision, the real MCP tool), the same way the
load-test script itself calls them, just at a size fast enough for the suite.
"""
from contextlib import closing

import pytest

from parking_chatbot import config
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.database import get_connection
from parking_chatbot.evaluation.load import summarize
from parking_chatbot.evaluation.load_scenarios import (
    admin_confirmation_scenario,
    chatbot_dialogue_scenario,
    mcp_recording_scenario,
)


@pytest.fixture
def capacity(sql_db):
    with closing(get_connection()) as conn, conn:
        conn.execute("DELETE FROM availability")
        conn.execute("INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, 50, 50)")


@pytest.fixture
def reservation_file(tmp_path, monkeypatch):
    path = tmp_path / "reservations.txt"
    monkeypatch.setattr(config, "MCP_RESERVATIONS_FILE_PATH", path)
    return path


# --- chatbot dialogue scenario -------------------------------------------------------------------


def test_chatbot_dialogue_scenario_runs_every_session_with_no_errors():
    result = chatbot_dialogue_scenario(sessions=4, concurrency=2)
    assert result.total_calls == 4
    assert result.error_rate == 0.0
    assert all(outcome.latency_seconds >= 0 for outcome in result.outcomes)


def test_chatbot_dialogue_scenario_honours_turns_per_session():
    """A session with more turns should, on average, take longer than one with fewer -
    proof that each call really drives turns_per_session turns, not a fixed number."""
    short = summarize(chatbot_dialogue_scenario(sessions=3, concurrency=3, turns_per_session=1))
    long = summarize(chatbot_dialogue_scenario(sessions=3, concurrency=3, turns_per_session=3))
    assert long["mean"] >= short["mean"]


# --- administrator confirmation scenario ---------------------------------------------------------


def test_admin_confirmation_scenario_decides_every_booking_with_no_errors(capacity):
    result = admin_confirmation_scenario(decisions=6, concurrency=3, approve_fraction=0.5)
    assert result.total_calls == 6
    assert result.error_rate == 0.0

    statuses = [row[0] for row in _all_statuses()]
    assert set(statuses) <= {"approved", "refused"}
    assert len(statuses) == 6


def test_admin_confirmation_scenario_does_not_call_the_real_recording_hook(capacity, monkeypatch):
    """Approving must not depend on (or fail because of) a real recording server."""
    called = []
    monkeypatch.setattr("parking_chatbot.recording.client.send_approved_booking", lambda booking_id: called.append(booking_id))

    admin_confirmation_scenario(decisions=4, concurrency=2, approve_fraction=1.0)

    assert called == []  # patched out for the duration of the scenario


def _all_statuses():
    with closing(get_connection()) as conn:
        return conn.execute("SELECT status FROM reservations").fetchall()


# --- MCP recording scenario -----------------------------------------------------------------------


def test_mcp_recording_scenario_records_every_booking_with_matching_file_lines(reservation_file):
    result, integrity = mcp_recording_scenario(bookings=5, concurrency=3)

    assert result.total_calls == 5
    assert result.error_rate == 0.0
    assert integrity["matches"] is True
    assert integrity["lines_written"] == 5
    assert integrity["file_ends_with_newline"] is True


def test_mcp_recording_scenario_starts_from_an_empty_file_even_if_one_exists(reservation_file):
    reservation_file.write_text("stale line from a previous run\n", encoding="utf-8")

    _, integrity = mcp_recording_scenario(bookings=3, concurrency=1)

    assert integrity["matches"] is True
    assert "stale line" not in reservation_file.read_text(encoding="utf-8")
