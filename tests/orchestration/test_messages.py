"""Tests for the user-facing wording the workflow produces.

Every unhappy path has to end in something clear to the user, so these tests
assert that each outcome names the booking and says plainly what happened.
"""
from parking_chatbot.orchestration import messages
from parking_chatbot.orchestration.state import initial_state

PASSES = [
    {"pass_id": "PK-AAAAAA-01", "car_number": "AB12CD", "vehicle_type": "car"},
    {"pass_id": "PK-AAAAAA-02", "car_number": "EF34GH", "vehicle_type": "three_wheeler"},
]


def _state(**overrides):
    state = initial_state("conv-1")
    state.update(overrides)
    return state


def test_format_passes_lists_one_line_per_vehicle():
    lines = messages.format_passes(PASSES).splitlines()
    assert len(lines) == 2
    assert "PK-AAAAAA-01" in lines[0] and "AB12CD" in lines[0]
    assert "three wheeler" in lines[1]  # underscores are not shown to the user


def test_format_passes_omits_a_missing_vehicle_type():
    line = messages.format_passes([{"pass_id": "PK-X-01", "car_number": "AB12CD", "vehicle_type": None}])
    assert line.strip() == "PK-X-01 - AB12CD"


def test_outcome_message_for_an_approved_and_recorded_booking_lists_the_passes():
    text = messages.outcome_message(
        _state(booking_id="PK-AAAAAA", approval_status="approved", recording_status="recorded", passes=PASSES)
    )
    assert "approved and recorded" in text
    assert "PK-AAAAAA-01" in text and "PK-AAAAAA-02" in text


def test_outcome_message_for_a_recording_failure_does_not_claim_it_was_recorded():
    text = messages.outcome_message(
        _state(booking_id="PK-AAAAAA", approval_status="approved", recording_status="failed", passes=PASSES)
    )
    assert "approved" in text
    assert "approved and recorded" not in text
    assert "PK-AAAAAA" in text


def test_outcome_message_for_a_refusal_includes_the_reason_when_there_is_one():
    with_reason = messages.outcome_message(
        _state(booking_id="PK-AAAAAA", approval_status="refused", decision_reason="level closed")
    )
    assert "refused" in with_reason and "level closed" in with_reason

    without_reason = messages.outcome_message(_state(booking_id="PK-AAAAAA", approval_status="refused"))
    assert "refused" in without_reason and "Reason:" not in without_reason


def test_outcome_message_for_a_timeout_says_the_booking_is_not_confirmed():
    text = messages.outcome_message(_state(booking_id="PK-AAAAAA", approval_status="expired"))
    assert "expired" in text and "not confirmed" in text


def test_outcome_message_falls_back_to_cancelled_then_submitted():
    assert messages.outcome_message(_state(cancelled=True)) == messages.CANCELLED_MESSAGE
    pending = messages.outcome_message(_state(booking_id="PK-AAAAAA", approval_status="pending"))
    assert "sent to an administrator" in pending
