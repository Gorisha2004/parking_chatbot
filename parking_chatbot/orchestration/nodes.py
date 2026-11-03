"""The workflow's nodes: thin wrappers around the Stage 1 to 3 components.

Each node does three things and no more - call one existing component, write what
it learned into the shared state, and (where the user needs to hear something)
add a notice. No retrieval, no reservation validation, no decision interpretation
and no file writing happens here; all of that already exists, behind the entry
points injected through WorkflowDependencies.

The dependencies are injected rather than imported at the point of use so every
node can be exercised against fakes - no LLM, no database, no network - which is
what the unit tests do (dependency inversion).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from parking_chatbot.chatbot import reservation_tools
from parking_chatbot.chatbot.agent import chat as default_chat
from parking_chatbot.db import admin as admin_db
from parking_chatbot.orchestration import messages
from parking_chatbot.orchestration.state import PassRecord, WorkflowState
from parking_chatbot.recording import client as recording_client

logger = logging.getLogger(__name__)

#: The Stage 1 chat entry point: (agent, user_message, thread_id) -> (answer, tools used).
ChatFn = Callable[[Any, str, str], tuple[str, list[str]]]
#: Reads one booking's current status and vehicles, or None if the ID is unknown.
ReadBookingFn = Callable[[str], dict | None]
#: Records one approved booking, raising RecordingError if it cannot.
RecordBookingFn = Callable[[str], bool]


class DraftReader(Protocol):
    """The read-only view the workflow needs of the Stage 1 reservation draft."""

    def __call__(self, thread_id: str) -> Any | None: ...


@dataclass(frozen=True)
class WorkflowDependencies:
    """Everything the nodes call out to, in one injectable place.

    The defaults wire up the real Stage 1 to 3 components, so production code
    builds this with nothing but the chat agent; tests pass fakes per field.
    """

    agent: Any
    chat: ChatFn = field(default=default_chat)
    read_draft: DraftReader = field(default=reservation_tools.get_draft)
    read_last_booking_id: Callable[[str], str | None] = field(
        default=reservation_tools.last_submitted_booking_id
    )
    read_booking: ReadBookingFn = field(default=admin_db.get_booking)
    record_booking: RecordBookingFn = field(default=recording_client.record_approved_booking)


def _passes_from(booking: dict) -> list[PassRecord]:
    """The booking's vehicles as the state carries them - one pass per vehicle."""
    return [
        PassRecord(
            pass_id=vehicle["pass_id"],
            car_number=vehicle["car_number"],
            vehicle_type=vehicle.get("vehicle_type"),
        )
        for vehicle in booking.get("vehicles", [])
    ]


def _with_notice(state: WorkflowState, notice: str) -> list[str]:
    """This turn's notices plus one more. Appended explicitly rather than through a
    reducer, so that starting a new turn can still clear the list outright."""
    return [*(state.get("notices") or []), notice]


class WorkflowNodes:
    """The node implementations, bound to one set of dependencies.

    A class rather than free functions taking a deps argument: LangGraph nodes must
    be single-argument callables, and bound methods give that without partials,
    while keeping the dependencies explicit and replaceable.
    """

    def __init__(self, deps: WorkflowDependencies) -> None:
        self._deps = deps

    # --- entry ----------------------------------------------------------------

    def ingest(self, state: WorkflowState) -> WorkflowState:
        """Open the turn: refresh the draft snapshot from the Stage 1 tools.

        Does no routing itself (that is routing.py's job) and calls nothing
        expensive, so a background tick costs one dictionary read.
        """
        draft = self._deps.read_draft(state["thread_id"])
        return WorkflowState(draft=draft.model_dump(mode="json") if draft else None)

    # --- user interaction (Stage 1) -------------------------------------------

    def user_interaction(self, state: WorkflowState) -> WorkflowState:
        """Answer the user by calling the Stage 1 guarded chat entry point.

        The agent behind it does the retrieval, the guardrails and the whole
        reservation collection; this node only reports what came back and reads
        the two facts the workflow has to route on - whether this turn submitted
        a reservation, and whether it discarded one.
        """
        thread_id = state["thread_id"]
        user_input = state.get("user_input", "")
        had_draft = self._deps.read_draft(thread_id) is not None

        answer, tools_used = self._deps.chat(self._deps.agent, user_input, thread_id)

        draft = self._deps.read_draft(thread_id)
        update = WorkflowState(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ],
            response=answer,
            tools_used=list(tools_used),
            draft=draft.model_dump(mode="json") if draft else None,
        )

        # A cancel only means "the user cancelled" if there really was a draft to
        # discard; the tool is also a harmless no-op the agent may call
        # defensively, and that must not look like an abandoned reservation.
        if "cancel_reservation" in tools_used and had_draft:
            update["cancelled"] = True
            return update

        if "submit_reservation" in tools_used:
            update.update(self._after_submit(state, thread_id))
        return update

    def _after_submit(self, state: WorkflowState, thread_id: str) -> WorkflowState:
        """Pick up the booking the Stage 1 submit tool just created.

        That tool returns its booking and pass IDs inside the tool message the LLM
        reads, so they are read back through the reservation tools' accessor and
        the booking record - never parsed out of the agent's prose.
        """
        booking_id = self._deps.read_last_booking_id(thread_id)
        if not booking_id:
            logger.warning("submit_reservation ran but no booking ID was recorded for thread %s", thread_id)
            return WorkflowState()
        booking = self._deps.read_booking(booking_id)
        if booking is None:
            logger.warning("Booking %s was submitted but cannot be read back", booking_id)
            return WorkflowState()
        return WorkflowState(
            booking_id=booking_id,
            passes=_passes_from(booking),
            approval_status="pending",
            decision_reason=None,
            recording_status="none",
            recording_error=None,
            cancelled=False,
            notices=_with_notice(state, messages.submitted_message(booking_id)),
        )

    # --- administrator approval (Stage 2) -------------------------------------

    def admin_approval(self, state: WorkflowState) -> WorkflowState:
        """Read how far the administrator has got with this booking.

        Read-only on purpose. Stage 2 keeps exactly one path that writes a
        decision (admin.agent.apply_decision, driven by the administrator's own
        message); the workflow observes that decision, it never makes one. It is
        also why this node does not block: the administrator is a person working
        in a different process, so the node checks, records what it found, and
        lets the turn end - the checkpointed state is what remembers that the
        booking is still pending.

        Expiry is not implemented here either: Stage 2 already expires an overdue
        booking lazily whenever it is read, so the timeout path arrives here as a
        plain 'expired' status.
        """
        booking_id = state.get("booking_id")
        if not booking_id:
            return WorkflowState()

        booking = self._deps.read_booking(booking_id)
        if booking is None:
            # Unknown ID: nothing will ever decide it, so treat it as a timeout
            # rather than leave the user waiting on a booking that cannot resolve.
            logger.warning("Booking %s is no longer readable; treating it as expired", booking_id)
            return WorkflowState(approval_status="expired")

        update = WorkflowState(
            approval_status=booking["status"],
            decision_reason=booking.get("decision_reason"),
        )
        passes = _passes_from(booking)
        if passes:
            update["passes"] = passes
        return update

    # --- recording (Stage 3) --------------------------------------------------

    def record(self, state: WorkflowState) -> WorkflowState:
        """Hand the approved booking to the Stage 3 recording client.

        Safe to reach more than once for the same booking: the approval hook on
        the administrator's side may already have recorded it, and both the client
        and the reservation file writer are idempotent per booking ID, so this
        node can never produce a duplicate or a partial set of lines.
        """
        booking_id = state.get("booking_id")
        if not booking_id:
            return WorkflowState()
        try:
            self._deps.record_booking(booking_id)
        except recording_client.RecordingError as error:
            logger.warning("Recording booking %s failed: %s", booking_id, error)
            return WorkflowState(recording_status="failed", recording_error=str(error))
        return WorkflowState(recording_status="recorded", recording_error=None)

    # --- terminal nodes -------------------------------------------------------

    def notify(self, state: WorkflowState) -> WorkflowState:
        """Turn the outcome already in the state into one clear sentence."""
        return WorkflowState(notices=_with_notice(state, messages.outcome_message(state)))

    def cancelled(self, state: WorkflowState) -> WorkflowState:
        """The user abandoned the reservation they were building.

        Clears the booking fields so the conversation goes back to having nothing
        in flight - no half-filled booking left behind in the checkpoint.
        """
        return WorkflowState(
            cancelled=True,
            draft=None,
            booking_id=None,
            passes=[],
            approval_status="none",
            decision_reason=None,
            recording_status="none",
            recording_error=None,
            notices=_with_notice(state, messages.CANCELLED_MESSAGE),
        )
