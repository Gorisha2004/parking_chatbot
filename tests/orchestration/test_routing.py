"""Tests for the conditional edges - one per branch, happy and unhappy.

The routers are pure functions of the state, so each branch is driven by a plain
dictionary: no graph, no nodes, no fakes needed.
"""
from langgraph.graph import END

from parking_chatbot.orchestration import routing
from parking_chatbot.orchestration.state import initial_state

BOOKING_ID = "PK-AAAAAA"


def _state(**overrides):
    state = initial_state("conv-1")
    state.update(overrides)
    return state


def _pending(**overrides):
    return _state(booking_id=BOOKING_ID, approval_status="pending", **overrides)


# --- route_turn ---------------------------------------------------------------


def test_a_user_message_goes_to_the_chat_agent():
    assert routing.route_turn(_state(turn_kind="message")) == routing.USER_INTERACTION


def test_a_user_message_still_goes_to_the_chat_agent_while_a_booking_is_pending():
    """The user's message must never be dropped in favour of checking a decision."""
    assert routing.route_turn(_pending(turn_kind="message")) == routing.USER_INTERACTION


def test_a_tick_with_a_pending_booking_goes_straight_to_the_approval_node():
    assert routing.route_turn(_pending(turn_kind="tick")) == routing.ADMIN_APPROVAL


def test_a_tick_with_nothing_pending_ends_the_turn_without_touching_the_llm():
    assert routing.route_turn(_state(turn_kind="tick")) == END
    assert routing.route_turn(_state(turn_kind="tick", booking_id=BOOKING_ID, approval_status="approved")) == END


# --- route_after_user_interaction ---------------------------------------------


def test_a_user_cancellation_goes_to_the_cancelled_node():
    assert routing.route_after_user_interaction(_state(cancelled=True)) == routing.CANCELLED


def test_a_cancellation_wins_over_a_booking_that_was_pending():
    assert routing.route_after_user_interaction(_pending(cancelled=True)) == routing.CANCELLED


def test_a_just_submitted_or_still_pending_booking_goes_to_the_approval_node():
    assert routing.route_after_user_interaction(_pending()) == routing.ADMIN_APPROVAL


def test_an_information_only_turn_ends_after_the_answer():
    assert routing.route_after_user_interaction(_state(response="We open at 6am.")) == END


def test_a_decided_booking_does_not_go_back_to_the_approval_node():
    """Once reported, a decided booking must not be re-announced every turn."""
    decided = _state(booking_id=BOOKING_ID, approval_status="approved", recording_status="recorded")
    assert routing.route_after_user_interaction(decided) == END


# --- route_after_approval -----------------------------------------------------


def test_an_approval_goes_to_the_recording_node():
    assert routing.route_after_approval(_state(approval_status="approved")) == routing.RECORD


def test_a_refusal_goes_straight_to_notify():
    assert routing.route_after_approval(_state(approval_status="refused")) == routing.NOTIFY


def test_a_refusal_with_a_reason_still_goes_to_notify():
    state = _state(approval_status="refused", decision_reason="level 3 is closed")
    assert routing.route_after_approval(state) == routing.NOTIFY


def test_a_timed_out_booking_goes_to_notify():
    assert routing.route_after_approval(_state(approval_status="expired")) == routing.NOTIFY


def test_a_still_pending_booking_ends_the_turn_so_the_user_can_keep_talking():
    assert routing.route_after_approval(_pending()) == END


def test_nothing_in_flight_ends_the_turn():
    assert routing.route_after_approval(_state(approval_status="none")) == END


def test_every_routing_destination_is_a_real_node_name():
    """Guards against a router returning a label the graph has no node for."""
    node_names = {
        routing.INGEST,
        routing.USER_INTERACTION,
        routing.ADMIN_APPROVAL,
        routing.RECORD,
        routing.NOTIFY,
        routing.CANCELLED,
    }
    destinations = {
        routing.route_turn(_state(turn_kind="message")),
        routing.route_turn(_pending(turn_kind="tick")),
        routing.route_after_user_interaction(_state(cancelled=True)),
        routing.route_after_user_interaction(_pending()),
        routing.route_after_approval(_state(approval_status="approved")),
        routing.route_after_approval(_state(approval_status="refused")),
    }
    assert destinations <= node_names
