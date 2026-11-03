"""The workflow's conditional edges: what runs next, given the state.

Routing is kept out of the nodes on purpose. A node reports what it found; these
functions decide where that leads. They are pure functions of the state - no I/O,
no side effects - which is what makes every branch, including each unhappy path,
testable on its own with a plain dictionary.

The unhappy paths and where they are decided:

* the user cancels     -> route_after_user_interaction -> CANCELLED
* the administrator refuses -> route_after_approval    -> NOTIFY
* the administrator times out (Stage 2 expires the booking lazily, so it arrives
  here as the 'expired' status) -> route_after_approval -> NOTIFY
* recording fails      -> the record node writes the failure to the state and the
  plain record -> notify edge reports it; the booking stays unrecorded so the
  Stage 3 retry sweep still picks it up.
"""
from langgraph.graph import END

from parking_chatbot.orchestration.state import WorkflowState, has_pending_booking

#: Node names, in one place so graph.py and these routers cannot drift apart.
INGEST = "ingest"
USER_INTERACTION = "user_interaction"
ADMIN_APPROVAL = "admin_approval"
RECORD = "record"
NOTIFY = "notify"
CANCELLED = "cancelled"


def route_turn(state: WorkflowState) -> str:
    """After ingest: answer the user, or just look for an administrator decision.

    A "tick" turn carries no user input. It exists so a decision made while the
    user was not typing still reaches them promptly, and it is skipped entirely
    when there is no booking in flight - a tick must never invoke the LLM.
    """
    if state.get("turn_kind") == "tick":
        return ADMIN_APPROVAL if has_pending_booking(state) else END
    return USER_INTERACTION


def route_after_user_interaction(state: WorkflowState) -> str:
    """After the chat agent has answered.

    Checking the approval after the answer, never before, is what keeps the user's
    message from being dropped on a turn where a decision also happens to land.
    """
    if state.get("cancelled"):
        return CANCELLED
    if has_pending_booking(state):
        return ADMIN_APPROVAL
    return END


def route_after_approval(state: WorkflowState) -> str:
    """After reading where the administrator got to.

    Still pending (or nothing in flight at all) ends the turn: the checkpointed
    state remembers the booking, so the next turn or tick picks it up again
    without the graph having to block on a person.
    """
    status = state.get("approval_status")
    if status == "approved":
        return RECORD
    if status in ("refused", "expired"):
        return NOTIFY
    return END
