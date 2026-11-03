"""Every user-facing sentence the workflow itself produces.

One module, so the wording is in one place and the nodes stay about control flow.
The chat agent's own answers are not here - those come from the LLM via Stage 1.

Each unhappy path has its own explicit message: the acceptance criteria require
that a refusal, a timeout, a recording failure and a cancellation all end in
something clear to the user, never in silence or a bare status code.
"""
from parking_chatbot.orchestration.state import PassRecord, WorkflowState

#: Shown when the administrator approves and the booking reaches the file.
APPROVED_TEMPLATE = (
    "Good news - booking {booking_id} has been approved and recorded. "
    "Show the pass ID for each vehicle on arrival:\n{passes}"
)
#: Shown when the administrator approves but the reservation file is not reachable.
#: Deliberately does not say "approved and recorded": the approval is real, the
#: file entry is not there yet, and the user is told exactly that rather than a
#: half-truth in either direction.
RECORDING_FAILED_TEMPLATE = (
    "Booking {booking_id} has been approved, but we could not write it to the "
    "reservation record just yet. Nothing is lost: it is queued and will be "
    "recorded automatically. Please keep your booking ID and check back with "
    "'what is the status of {booking_id}?'."
)
REFUSED_TEMPLATE = "Sorry - booking {booking_id} was refused by the administrator."
REFUSED_WITH_REASON_TEMPLATE = REFUSED_TEMPLATE + " Reason: {reason}"
EXPIRED_TEMPLATE = (
    "Booking {booking_id} expired before an administrator could review it, so it "
    "is not confirmed. You are welcome to make a new reservation."
)
SUBMITTED_TEMPLATE = (
    "Booking {booking_id} has been sent to an administrator for approval. It is "
    "not confirmed yet - you can keep asking me anything in the meantime, and I "
    "will tell you as soon as there is a decision."
)
CANCELLED_MESSAGE = (
    "The reservation you were building has been discarded. Nothing was submitted."
)


def format_passes(passes: list[PassRecord]) -> str:
    """One line per vehicle, matching the one-line-per-vehicle reservation file."""
    return "\n".join(
        f"  {record['pass_id']} - {record['car_number']}"
        + (f" ({record['vehicle_type'].replace('_', ' ')})" if record.get("vehicle_type") else "")
        for record in passes
    )


def submitted_message(booking_id: str) -> str:
    return SUBMITTED_TEMPLATE.format(booking_id=booking_id)


def approved_message(booking_id: str, passes: list[PassRecord]) -> str:
    return APPROVED_TEMPLATE.format(booking_id=booking_id, passes=format_passes(passes))


def recording_failed_message(booking_id: str) -> str:
    return RECORDING_FAILED_TEMPLATE.format(booking_id=booking_id)


def refused_message(booking_id: str, reason: str | None) -> str:
    if reason:
        return REFUSED_WITH_REASON_TEMPLATE.format(booking_id=booking_id, reason=reason)
    return REFUSED_TEMPLATE.format(booking_id=booking_id)


def expired_message(booking_id: str) -> str:
    return EXPIRED_TEMPLATE.format(booking_id=booking_id)


def outcome_message(state: WorkflowState) -> str:
    """The one message that describes how this booking ended, chosen from the
    state the approval and recording nodes already wrote.

    Centralised here so the notify node stays a single call and no node has to
    know the wording of an outcome it did not produce.
    """
    booking_id = state.get("booking_id") or "your booking"
    status = state.get("approval_status")
    if status == "approved":
        if state.get("recording_status") == "recorded":
            return approved_message(booking_id, state.get("passes") or [])
        return recording_failed_message(booking_id)
    if status == "refused":
        return refused_message(booking_id, state.get("decision_reason"))
    if status == "expired":
        return expired_message(booking_id)
    if state.get("cancelled"):
        return CANCELLED_MESSAGE
    return submitted_message(booking_id)
