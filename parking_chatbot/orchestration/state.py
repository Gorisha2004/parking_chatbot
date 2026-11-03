"""The one typed shared state every node in the workflow reads and writes.

Kept deliberately flat and JSON-serializable: the whole thing is checkpointed
after every node, and a checkpoint that cannot be serialized is a checkpoint
that cannot survive the process restart it exists to survive. That is also why
the collected reservation is mirrored in as a plain dict (`draft`) rather than
as a ReservationDraft - the domain model stays the single source of truth in
reservation.models, this is only a snapshot of it for the workflow to route on.
"""
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

#: Where a booking stands with the administrator. "none" means this conversation
#: has no booking in flight at all - the normal state for an information-only chat.
ApprovalStatus = Literal["none", "pending", "approved", "refused", "expired"]

#: Whether the approved booking has reached the reservation file. "none" means
#: recording has not been attempted (nothing was approved, or not yet).
RecordingStatus = Literal["none", "recorded", "failed"]

#: What the turn being processed is. A "message" turn carries the user's input; a
#: "tick" turn carries none and exists only so the graph can pick up an
#: administrator decision that landed while the user was not typing.
TurnKind = Literal["message", "tick"]

TERMINAL_APPROVAL_STATUSES: tuple[ApprovalStatus, ...] = ("approved", "refused", "expired")


class PassRecord(TypedDict):
    """One vehicle's pass in a submitted booking, as the workflow carries it."""

    pass_id: str
    car_number: str
    vehicle_type: str | None


class WorkflowState(TypedDict, total=False):
    """The shared state of one conversation's workflow.

    `total=False` because a node returns only the keys it actually changed:
    LangGraph merges that partial update into the checkpointed state, so no node
    has to know or restate anything it is not responsible for (single
    responsibility, and no accidental clobbering of another node's fields).
    """

    # --- conversation ---------------------------------------------------------
    #: The conversation transcript. `add_messages` is LangGraph's own reducer, so
    #: a node appends by returning just the new messages.
    messages: Annotated[list, add_messages]
    #: Identifies this conversation; also the checkpointer's thread id, and the
    #: key the Stage 1 reservation tools store their draft under.
    thread_id: str
    turn_kind: TurnKind
    #: The user's input for a "message" turn; empty on a "tick".
    user_input: str
    #: What the chat agent answered this turn, if it ran at all.
    response: str
    #: Names of the Stage 1 tools the agent used this turn - how the workflow
    #: sees that a reservation was submitted or cancelled.
    tools_used: list[str]

    # --- the collected reservation --------------------------------------------
    #: Snapshot of the in-progress reservation draft, or None once it is submitted.
    draft: dict | None
    #: The booking this conversation is waiting on, once submitted.
    booking_id: str | None
    #: One pass per vehicle in that booking (a group booking has several).
    passes: list[PassRecord]

    # --- the administrator's decision -----------------------------------------
    approval_status: ApprovalStatus
    decision_reason: str | None

    # --- recording ------------------------------------------------------------
    recording_status: RecordingStatus
    recording_error: str | None

    # --- outcome --------------------------------------------------------------
    #: True when the user abandoned the reservation they were building.
    cancelled: bool
    #: Everything to print to the user at the end of this turn, in order. Replaced
    #: (not accumulated) each turn: it is this turn's output, not a history.
    notices: list[str]


def initial_state(thread_id: str) -> WorkflowState:
    """A fresh conversation's state, with every field at its empty value.

    Spelled out rather than left to `total=False` defaults so that nodes, routing
    and tests all read the same explicit starting point instead of each guessing
    what a missing key means.
    """
    return WorkflowState(
        messages=[],
        thread_id=thread_id,
        turn_kind="message",
        user_input="",
        response="",
        tools_used=[],
        draft=None,
        booking_id=None,
        passes=[],
        approval_status="none",
        decision_reason=None,
        recording_status="none",
        recording_error=None,
        cancelled=False,
        notices=[],
    )


def turn_input(user_input: str, *, turn_kind: TurnKind = "message") -> WorkflowState:
    """The partial state that starts one turn.

    Clears the previous turn's per-turn fields so a stale answer or notice can
    never be shown twice, while leaving the durable fields (the booking being
    waited on, its passes, the approval status) untouched for the checkpointed
    state to carry forward.
    """
    return WorkflowState(
        turn_kind=turn_kind,
        user_input=user_input,
        response="",
        tools_used=[],
        notices=[],
    )


def has_pending_booking(state: WorkflowState) -> bool:
    """Whether this conversation is waiting on an administrator decision."""
    return bool(state.get("booking_id")) and state.get("approval_status") == "pending"


def is_awaiting_recording(state: WorkflowState) -> bool:
    """Whether an approved booking still has to reach the reservation file."""
    return state.get("approval_status") == "approved" and state.get("recording_status") != "recorded"
