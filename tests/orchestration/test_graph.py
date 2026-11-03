"""Tests for the compiled graph: wiring, one-turn behaviour, and persistence.

Everything runs against the injected fakes - no LLM, no admin channel, no
recording server - so these exercise the graph itself, not its components.
"""
from langgraph.graph import END

from parking_chatbot.orchestration import routing
from parking_chatbot.orchestration.checkpointing import build_checkpointer, close_checkpointer
from parking_chatbot.orchestration.graph import build_workflow, run_turn, workflow_state


def _workflow(fakes, checkpointer=None):
    return build_workflow(deps=fakes.deps(), checkpointer=checkpointer or build_checkpointer(backend="memory"))


def _submit(workflow, fakes, thread_id="conv-1"):
    """Drive one turn that submits a reservation, and leave it pending."""
    fakes.chat.tools = ["show_reservation_summary", "submit_reservation"]
    state = run_turn(workflow, thread_id, "yes, confirm")
    fakes.chat.tools = []
    return state


# --- wiring -------------------------------------------------------------------


def test_the_graph_contains_every_node_the_routers_can_return(fakes):
    nodes = set(_workflow(fakes).get_graph().nodes)
    assert {
        routing.INGEST,
        routing.USER_INTERACTION,
        routing.ADMIN_APPROVAL,
        routing.RECORD,
        routing.NOTIFY,
        routing.CANCELLED,
    } <= nodes


def test_an_information_only_turn_answers_and_stops(fakes):
    fakes.chat.answer = "We open at 6am."
    state = run_turn(_workflow(fakes), "conv-1", "when do you open?")
    assert state["response"] == "We open at 6am."
    assert state["notices"] == []
    assert state["approval_status"] == "none"
    assert fakes.bookings.reads == []  # the approval node was never reached


# --- the happy path -----------------------------------------------------------


def test_submitting_leaves_the_booking_pending_and_tells_the_user(fakes):
    state = _submit(_workflow(fakes), fakes)
    assert state["approval_status"] == "pending"
    assert state["booking_id"] == fakes.booking_id
    assert len(state["passes"]) == 2
    assert "sent to an administrator" in state["notices"][-1]


def test_an_approval_is_recorded_and_confirmed_on_the_next_tick(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.bookings.status = "approved"
    state = run_turn(workflow, "conv-1", "", turn_kind="tick")

    assert state["recording_status"] == "recorded"
    assert fakes.recorder.calls == [fakes.booking_id]
    notice = state["notices"][-1]
    assert "approved and recorded" in notice
    # One line per vehicle, as the reservation file has.
    assert f"{fakes.booking_id}-01" in notice and f"{fakes.booking_id}-02" in notice


def test_a_decision_that_lands_during_a_user_message_is_reported_with_the_answer(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.bookings.status = "approved"
    fakes.chat.answer = "We have 40 free spaces."
    state = run_turn(workflow, "conv-1", "how many spaces are free?")

    assert state["response"] == "We have 40 free spaces."  # the question was still answered
    assert "approved and recorded" in state["notices"][-1]


# --- the unhappy paths --------------------------------------------------------


def test_a_refusal_ends_in_a_clear_message_and_nothing_is_recorded(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.bookings.status = "refused"
    fakes.bookings.decision_reason = "level 3 is closed"
    state = run_turn(workflow, "conv-1", "", turn_kind="tick")

    assert state["approval_status"] == "refused"
    assert state["recording_status"] == "none"
    assert fakes.recorder.calls == []  # no partial file entry
    assert "refused" in state["notices"][-1] and "level 3 is closed" in state["notices"][-1]


def test_a_timeout_ends_in_a_clear_message_and_nothing_is_recorded(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.bookings.status = "expired"
    state = run_turn(workflow, "conv-1", "", turn_kind="tick")

    assert state["approval_status"] == "expired"
    assert fakes.recorder.calls == []
    assert "expired" in state["notices"][-1] and "not confirmed" in state["notices"][-1]


def test_a_recording_failure_is_reported_without_claiming_success(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.bookings.status = "approved"
    fakes.recorder.fail = True
    state = run_turn(workflow, "conv-1", "", turn_kind="tick")

    assert state["approval_status"] == "approved"
    assert state["recording_status"] == "failed"
    assert state["recording_error"]
    notice = state["notices"][-1]
    assert "approved" in notice and "approved and recorded" not in notice
    assert "will be recorded automatically" in notice


def test_a_user_cancellation_clears_the_booking_and_says_so(fakes):
    workflow = _workflow(fakes)
    fakes.draft = fakes.make_draft()
    fakes.chat.tools = ["cancel_reservation"]
    state = run_turn(workflow, "conv-1", "forget it")

    assert state["cancelled"] is True
    assert state["booking_id"] is None
    assert state["approval_status"] == "none"
    assert state["passes"] == []
    assert "discarded" in state["notices"][-1]


def test_a_cancellation_leaves_no_pending_booking_behind_in_the_checkpoint(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)
    fakes.draft = fakes.make_draft()
    fakes.chat.tools = ["cancel_reservation"]
    run_turn(workflow, "conv-1", "cancel that")

    persisted = workflow_state(workflow, "conv-1")
    assert persisted["booking_id"] is None
    assert persisted["approval_status"] == "none"
    assert routing.route_after_approval(persisted) == END


# --- persistence --------------------------------------------------------------


def test_a_pending_approval_survives_a_fresh_graph_on_the_same_sqlite_file(fakes, tmp_path):
    """The point of checkpointing: the administrator can take as long as they
    like, and a restarted process still knows which booking it is waiting on."""
    db_path = tmp_path / "checkpoints.sqlite"

    first = build_checkpointer(backend="sqlite", db_path=db_path)
    try:
        _submit(_workflow(fakes, first), fakes)
    finally:
        close_checkpointer(first)

    # A completely new checkpointer, graph and nodes - only the file is shared.
    second = build_checkpointer(backend="sqlite", db_path=db_path)
    try:
        resumed = _workflow(fakes, second)
        persisted = workflow_state(resumed, "conv-1")
        assert persisted["booking_id"] == fakes.booking_id
        assert persisted["approval_status"] == "pending"
        assert len(persisted["passes"]) == 2

        # And the pending booking can still be carried to its decision.
        fakes.bookings.status = "approved"
        state = run_turn(resumed, "conv-1", "", turn_kind="tick")
        assert state["recording_status"] == "recorded"
    finally:
        close_checkpointer(second)


def test_the_conversation_continues_while_a_booking_is_pending(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)

    fakes.chat.answer = "Level 2 has space."
    state = run_turn(workflow, "conv-1", "is level 2 free?")

    assert state["response"] == "Level 2 has space."
    assert state["approval_status"] == "pending"  # still waiting, still remembered
    assert state["booking_id"] == fakes.booking_id


def test_each_turn_starts_with_a_clean_slate_of_notices(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes)
    state = run_turn(workflow, "conv-1", "thanks")
    assert state["notices"] == []  # the submitted notice is not repeated


def test_two_conversations_on_one_graph_do_not_see_each_other(fakes):
    workflow = _workflow(fakes)
    _submit(workflow, fakes, thread_id="conv-1")
    run_turn(workflow, "conv-2", "hello")

    assert workflow_state(workflow, "conv-2")["booking_id"] is None
    assert workflow_state(workflow, "conv-1")["booking_id"] == fakes.booking_id
