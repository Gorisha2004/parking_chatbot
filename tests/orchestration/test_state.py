"""Tests for the workflow's typed shared state."""
from parking_chatbot.orchestration.state import (
    WorkflowState,
    has_pending_booking,
    initial_state,
    is_awaiting_recording,
    turn_input,
)

EXPECTED_FIELDS = set(WorkflowState.__annotations__)


def test_initial_state_fills_every_declared_field():
    state = initial_state("conv-1")
    assert set(state) == EXPECTED_FIELDS
    assert state["thread_id"] == "conv-1"
    assert state["approval_status"] == "none"
    assert state["recording_status"] == "none"
    assert state["booking_id"] is None
    assert state["cancelled"] is False


def test_turn_input_clears_per_turn_fields_and_touches_nothing_durable():
    update = turn_input("what are your prices?")
    assert update["user_input"] == "what are your prices?"
    assert update["turn_kind"] == "message"
    assert update["response"] == ""
    assert update["notices"] == []
    assert update["tools_used"] == []
    # A new turn must not reset the booking being waited on, or its decision.
    assert not {"booking_id", "approval_status", "passes", "recording_status"} & set(update)


def test_turn_input_supports_a_tick_turn_with_no_user_text():
    update = turn_input("", turn_kind="tick")
    assert update["turn_kind"] == "tick"
    assert update["user_input"] == ""


def test_has_pending_booking_needs_both_an_id_and_a_pending_status():
    assert has_pending_booking({"booking_id": "PK-AAAAAA", "approval_status": "pending"})
    assert not has_pending_booking({"booking_id": "PK-AAAAAA", "approval_status": "approved"})
    assert not has_pending_booking({"booking_id": None, "approval_status": "pending"})
    assert not has_pending_booking(initial_state("conv-1"))


def test_is_awaiting_recording_only_while_an_approval_is_unrecorded():
    assert is_awaiting_recording({"approval_status": "approved", "recording_status": "none"})
    assert is_awaiting_recording({"approval_status": "approved", "recording_status": "failed"})
    assert not is_awaiting_recording({"approval_status": "approved", "recording_status": "recorded"})
    assert not is_awaiting_recording({"approval_status": "refused", "recording_status": "none"})
