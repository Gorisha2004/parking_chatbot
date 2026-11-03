"""Integration tests of the full LangGraph workflow: Stage 1 reservation
collection, Stage 2 administrator decisions and Stage 3 file recording, wired
together exactly as orchestration.graph.build_workflow wires them for the real
CLI - with only the LLM and the recording network hop faked (see conftest.py).

One test per path named in stage4.md's integration-testing requirement -
approved, refused, timeout, recording failure - plus the user-cancels path
named alongside them in the routing requirement. Each asserts both the user
message (acceptance criterion 2: every unhappy path ends in a clear message)
and the on-disk state (criterion 2: no inconsistent state or partial entry).
"""
from contextlib import closing
from datetime import datetime, timedelta, timezone

from parking_chatbot import config
from parking_chatbot.admin.agent import apply_decision
from parking_chatbot.chatbot import reservation_tools as rt
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.database import get_connection
from parking_chatbot.orchestration.graph import run_turn


def _future_times() -> tuple[str, str]:
    start = datetime.now() + timedelta(days=1)
    end = start + timedelta(hours=2)
    return start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")


def _booker_and_vehicles_steps(plates: tuple[str, ...]) -> list[tuple]:
    """The three real tool calls that collect and confirm a group booking."""
    start, end = _future_times()
    return [
        (rt.update_reservation_details, {"first_name": "anna", "last_name": "smith", "start": start, "end": end}),
        (rt.add_vehicles, {"vehicles": [{"car_number": plate, "vehicle_type": "car"} for plate in plates]}),
        (rt.show_reservation_summary, {}),
    ]


def _submit(real_tool_chat, workflow_factory, thread_id: str, plates: tuple[str, ...] = ("AB123CD", "CD456EF")):
    """Run the turns that build and submit a group booking.

    Returns (booking_id, workflow, chat) - the chat is returned too so a test
    that needs to keep talking in the same conversation can push more scripted
    steps onto it and keep driving the same workflow instance (and so the same
    checkpointer, the same thread), rather than building a second graph that
    would start that thread id from an empty checkpoint.
    """
    steps = _booker_and_vehicles_steps(plates) + [(rt.submit_reservation, {})]
    chat = real_tool_chat(steps=steps)
    workflow = workflow_factory(chat)
    state = None
    # Not `for _ in steps`: RealToolChat.__call__ pops from this same list each
    # turn, so iterating the list directly would skip steps as it shrinks.
    for _ in range(len(steps)):
        state = run_turn(workflow, thread_id, "irrelevant - RealToolChat ignores the text")
    assert state["approval_status"] == "pending"
    return state["booking_id"], workflow, chat


def _file_lines_for(path, booking_id: str) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if f"{booking_id}-" in line]


def _backdate_booking(booking_id: str, minutes_ago: int) -> None:
    moment = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
    with closing(get_connection()) as conn, conn:
        conn.execute("UPDATE reservations SET created_at = ? WHERE booking_id = ?", (moment, booking_id))


# --- approved path (also covers the group-booking acceptance criterion) -----------------------


def test_approved_booking_is_recorded_with_one_line_per_vehicle(
    real_tool_chat, workflow, thread_id, capacity, real_recording_server, reservation_file
):
    booking_id, wf, _chat = _submit(real_tool_chat, workflow, thread_id, plates=("AB123CD", "CD456EF"))

    apply_decision(booking_id, "approved", reason=None)  # the real Stage 2 write path
    state = run_turn(wf, thread_id, "", turn_kind="tick")

    assert state["approval_status"] == "approved"
    assert state["recording_status"] == "recorded"
    assert "approved and recorded" in state["notices"][-1]

    lines = _file_lines_for(reservation_file, booking_id)
    assert len(lines) == 2  # one per vehicle, per the reservation file's format
    assert all(booking_id in line and " | " in line for line in lines)
    # Field order is `Name | Car Number | Period | Approval Time | Pass ID | Vehicle Type`.
    plates_in_file = {line.split(" | ")[1] for line in lines}
    assert plates_in_file == {"AB123CD", "CD456EF"}


# --- refused path ---------------------------------------------------------------------------


def test_refused_booking_ends_in_a_clear_message_and_nothing_is_written(
    real_tool_chat, workflow, thread_id, capacity, real_recording_server, reservation_file
):
    booking_id, wf, _chat = _submit(real_tool_chat, workflow, thread_id)

    apply_decision(booking_id, "refused", reason="level 3 is closed for maintenance")
    state = run_turn(wf, thread_id, "", turn_kind="tick")

    assert state["approval_status"] == "refused"
    assert state["recording_status"] == "none"
    notice = state["notices"][-1]
    assert "refused" in notice and "level 3 is closed for maintenance" in notice
    assert _file_lines_for(reservation_file, booking_id) == []


# --- timeout path (administrator never answers) ---------------------------------------------


def test_an_overdue_booking_expires_and_ends_in_a_clear_message(
    real_tool_chat, workflow, thread_id, capacity, real_recording_server, reservation_file, monkeypatch
):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    booking_id, wf, _chat = _submit(real_tool_chat, workflow, thread_id)
    _backdate_booking(booking_id, minutes_ago=61)  # as if submitted, then left unanswered

    state = run_turn(wf, thread_id, "", turn_kind="tick")

    assert state["approval_status"] == "expired"
    assert state["recording_status"] == "none"
    notice = state["notices"][-1]
    assert "expired" in notice and "not confirmed" in notice
    assert _file_lines_for(reservation_file, booking_id) == []
    # The expiry is persisted by Stage 2's lazy check, not just reported once.
    assert admin_db.get_booking(booking_id)["status"] == "expired"


# --- recording failure path --------------------------------------------------------------------


def test_a_recording_failure_is_reported_and_leaves_no_partial_entry(
    real_tool_chat, workflow, thread_id, capacity, broken_recording_server, reservation_file
):
    booking_id, wf, _chat = _submit(real_tool_chat, workflow, thread_id)

    apply_decision(booking_id, "approved", reason=None)  # best-effort hook fails silently here too
    state = run_turn(wf, thread_id, "", turn_kind="tick")

    assert state["approval_status"] == "approved"
    assert state["recording_status"] == "failed"
    assert state["recording_error"]
    notice = state["notices"][-1]
    assert "approved" in notice and "approved and recorded" not in notice
    assert "will be recorded automatically" in notice
    assert _file_lines_for(reservation_file, booking_id) == []  # no partial entry

    # The booking is still visible to the Stage 3 retry sweep - nothing was lost.
    assert booking_id in [b.booking_id for b in admin_db.list_unrecorded_approved_bookings()]


def test_a_recording_failure_does_not_block_the_user_from_asking_something_else(
    real_tool_chat, workflow, thread_id, capacity, broken_recording_server, reservation_file
):
    booking_id, wf, chat = _submit(real_tool_chat, workflow, thread_id)
    apply_decision(booking_id, "approved", reason=None)
    run_turn(wf, thread_id, "", turn_kind="tick")  # the failure is reported once

    # Keep talking in the same conversation: push one more real tool call onto
    # the same scripted chat and drive the same workflow/checkpointer instance.
    chat.steps.append((rt.check_booking_status, {"booking_id": booking_id}))
    state = run_turn(wf, thread_id, "what is the status of my booking?")

    assert "approved" in state["response"]
    assert state["notices"] == []  # the failure is not re-announced every turn


# --- user cancels (the remaining unhappy path named in the routing requirement) -----------------


def test_a_user_cancellation_discards_the_draft_and_nothing_is_ever_submitted(
    real_tool_chat, workflow, thread_id, capacity, real_recording_server, reservation_file
):
    start, end = _future_times()
    steps = [
        (rt.update_reservation_details, {"first_name": "anna", "last_name": "smith", "start": start, "end": end}),
        (rt.add_vehicles, {"vehicles": [{"car_number": "AB123CD", "vehicle_type": "car"}]}),
        (rt.cancel_reservation, {}),
    ]
    chat = real_tool_chat(steps=steps)
    wf = workflow(chat)

    state = None
    for _ in range(len(steps)):
        state = run_turn(wf, thread_id, "irrelevant")

    assert state["cancelled"] is True
    assert state["booking_id"] is None
    assert state["approval_status"] == "none"
    assert "discarded" in state["notices"][-1]
    assert rt.get_draft(thread_id) is None  # the real Stage 1 draft is really gone
    assert reservation_file.exists() is False  # nothing was ever submitted, let alone recorded
