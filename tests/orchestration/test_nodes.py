"""Tests for each workflow node, against fakes for every dependency."""
import pytest

from parking_chatbot.orchestration import messages
from parking_chatbot.orchestration.nodes import WorkflowNodes
from parking_chatbot.orchestration.state import initial_state

BOOKING_ID = "PK-AAAAAA"


def _nodes(fakes) -> WorkflowNodes:
    return WorkflowNodes(fakes.deps())


def _state(**overrides):
    state = initial_state("conv-1")
    state.update(overrides)
    return state


# --- ingest -------------------------------------------------------------------


def test_ingest_snapshots_an_existing_draft(fakes):
    fakes.draft = fakes.make_draft({"first_name": "Anna", "vehicles": []})
    update = _nodes(fakes).ingest(_state())
    assert update["draft"] == {"first_name": "Anna", "vehicles": []}


def test_ingest_reports_no_draft_when_nothing_is_being_collected(fakes):
    assert _nodes(fakes).ingest(_state())["draft"] is None


# --- user interaction ---------------------------------------------------------


def test_user_interaction_records_the_answer_and_the_transcript(fakes):
    fakes.chat.answer = "We open at 6am."
    update = _nodes(fakes).user_interaction(_state(user_input="when do you open?"))
    assert update["response"] == "We open at 6am."
    assert [m["role"] for m in update["messages"]] == ["user", "assistant"]
    assert fakes.chat.calls[0][1:] == ("when do you open?", "conv-1")


def test_user_interaction_picks_up_a_submitted_booking_and_its_passes(fakes):
    fakes.chat.tools = ["show_reservation_summary", "submit_reservation"]
    update = _nodes(fakes).user_interaction(_state(user_input="yes, confirm"))
    assert update["booking_id"] == BOOKING_ID
    assert update["approval_status"] == "pending"
    assert [p["pass_id"] for p in update["passes"]] == [f"{BOOKING_ID}-01", f"{BOOKING_ID}-02"]
    assert update["notices"] == [messages.submitted_message(BOOKING_ID)]


def test_user_interaction_treats_a_cancel_with_a_draft_as_a_cancellation(fakes):
    fakes.draft = fakes.make_draft()
    fakes.chat.tools = ["cancel_reservation"]
    update = _nodes(fakes).user_interaction(_state(user_input="forget it"))
    assert update["cancelled"] is True


def test_user_interaction_ignores_a_cancel_that_had_no_draft_to_discard(fakes):
    fakes.draft = None
    fakes.chat.tools = ["cancel_reservation"]
    update = _nodes(fakes).user_interaction(_state(user_input="forget it"))
    assert "cancelled" not in update


def test_user_interaction_survives_a_submit_whose_booking_cannot_be_read_back(fakes):
    fakes.chat.tools = ["submit_reservation"]
    fakes.bookings.known = False
    update = _nodes(fakes).user_interaction(_state(user_input="yes"))
    assert "booking_id" not in update  # no half-filled booking is written to the state
    assert update["response"]


def test_user_interaction_survives_a_submit_with_no_booking_id_on_record(fakes):
    fakes.chat.tools = ["submit_reservation"]
    fakes.last_booking_id = None
    update = _nodes(fakes).user_interaction(_state(user_input="yes"))
    assert "booking_id" not in update


# --- administrator approval ---------------------------------------------------


@pytest.mark.parametrize("status", ["pending", "approved", "refused", "expired"])
def test_admin_approval_reports_the_status_it_reads(fakes, status):
    fakes.bookings.status = status
    update = _nodes(fakes).admin_approval(_state(booking_id=BOOKING_ID, approval_status="pending"))
    assert update["approval_status"] == status
    assert fakes.bookings.reads == [BOOKING_ID]


def test_admin_approval_carries_the_refusal_reason(fakes):
    fakes.bookings.status = "refused"
    fakes.bookings.decision_reason = "level 3 is closed"
    update = _nodes(fakes).admin_approval(_state(booking_id=BOOKING_ID))
    assert update["decision_reason"] == "level 3 is closed"


def test_admin_approval_treats_an_unreadable_booking_as_a_timeout(fakes):
    fakes.bookings.known = False
    update = _nodes(fakes).admin_approval(_state(booking_id=BOOKING_ID))
    assert update["approval_status"] == "expired"


def test_admin_approval_does_nothing_without_a_booking(fakes):
    assert _nodes(fakes).admin_approval(_state()) == {}
    assert fakes.bookings.reads == []


def test_admin_approval_never_writes_a_decision(fakes):
    """The workflow observes the administrator's decision; it must not make one."""
    nodes = _nodes(fakes)
    nodes.admin_approval(_state(booking_id=BOOKING_ID))
    # The only dependency touched is the read-only booking lookup.
    assert fakes.recorder.calls == []
    assert fakes.chat.calls == []


# --- recording ----------------------------------------------------------------


def test_record_marks_the_booking_recorded_on_success(fakes):
    update = _nodes(fakes).record(_state(booking_id=BOOKING_ID, approval_status="approved"))
    assert update == {"recording_status": "recorded", "recording_error": None}
    assert fakes.recorder.calls == [BOOKING_ID]


def test_record_reports_a_failure_instead_of_swallowing_it(fakes):
    fakes.recorder.fail = True
    update = _nodes(fakes).record(_state(booking_id=BOOKING_ID, approval_status="approved"))
    assert update["recording_status"] == "failed"
    assert BOOKING_ID in update["recording_error"]


def test_record_counts_an_already_recorded_booking_as_recorded(fakes):
    """The approval hook on the administrator's side may have got there first."""
    fakes.recorder.already_recorded = True
    update = _nodes(fakes).record(_state(booking_id=BOOKING_ID, approval_status="approved"))
    assert update["recording_status"] == "recorded"


def test_record_does_nothing_without_a_booking(fakes):
    assert _nodes(fakes).record(_state()) == {}
    assert fakes.recorder.calls == []


# --- terminal nodes -----------------------------------------------------------


def test_notify_appends_the_outcome_to_this_turns_notices(fakes):
    state = _state(booking_id=BOOKING_ID, approval_status="refused", notices=["earlier notice"])
    update = _nodes(fakes).notify(state)
    assert update["notices"][0] == "earlier notice"
    assert "refused" in update["notices"][1]


def test_cancelled_clears_the_booking_fields_and_explains_itself(fakes):
    state = _state(booking_id=BOOKING_ID, approval_status="pending", draft={"first_name": "Anna"})
    update = _nodes(fakes).cancelled(state)
    assert update["booking_id"] is None
    assert update["draft"] is None
    assert update["passes"] == []
    assert update["approval_status"] == "none"
    assert update["notices"] == [messages.CANCELLED_MESSAGE]
