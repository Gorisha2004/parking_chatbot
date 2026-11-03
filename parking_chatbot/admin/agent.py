"""The second agent: a LangChain tool-calling agent the administrator chats with to
review and decide on escalated bookings.

The decision-writing path is deliberately NOT left to the LLM. Confirming or creating
a decision proposal from the administrator's raw message is handled by deterministic
code (update_proposal, apply_decision) that runs before the agent is ever invoked -
see admin_chat(). The agent itself only handles browsing (listing bookings, showing
details) and open-ended conversation; its one decision-writing tool (record_decision)
exists as a defense-in-depth fallback, and re-derives the decision from the
administrator's actual latest message (never from an LLM-supplied copy of it) before
writing anything.

Interpreting a message checks unambiguous keywords first, deterministically - no LLM
call, and no room for model error, on the common case. An LLM is used only for
genuinely unclear text reaching the REST API, and even then it may only conclude
"refused" or give up - approval is a value it cannot produce. Ambiguity must never
approve.
"""
import json
import re
from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from parking_chatbot.admin.models import AdminDecision, ApprovalRequest
from parking_chatbot.db import admin as admin_db
from parking_chatbot.llm import get_chat_model
from parking_chatbot.reservation.ids import BOOKING_ID_LENGTH, BOOKING_ID_PREFIX, ID_ALPHABET

# "not/never approve", "don't confirm" etc. - checked first so a negated approval is
# never misread as a plain one.
_NEGATED_APPROVAL = re.compile(
    r"\b(not|never|don'?t|doesn'?t|do\s+not|does\s+not)\b[\w\s]{0,15}\b(approve[ds]?|yes|confirm(ed)?|ok(ay)?)\b",
    re.IGNORECASE,
)
_APPROVE = re.compile(r"\b(approve[ds]?|yes|confirm(ed)?|ok(ay)?)\b", re.IGNORECASE)
_REFUSE = re.compile(r"\b(refuse[ds]?|reject(ed)?|deny|denied|no)\b", re.IGNORECASE)

# Anywhere in a sentence ("approve PK-ABCDEF please") - unlike the chatbot's anchored
# BOOKING_ID_PATTERN, which validates a whole string given as an argument.
_BOOKING_ID_SEARCH = re.compile(rf"\b{BOOKING_ID_PREFIX}-[{ID_ALPHABET}]{{{BOOKING_ID_LENGTH}}}\b", re.IGNORECASE)
_BOOKING_ID_FULLMATCH = re.compile(rf"^{BOOKING_ID_PREFIX}-[{ID_ALPHABET}]{{{BOOKING_ID_LENGTH}}}$")

# "is it approved?", "has PK-X been refused yet" etc. - past-participle forms like
# "approved"/"refused" match _APPROVE/_REFUSE below, so a plain status QUESTION would
# otherwise be misread as a decision COMMAND. Checked first so a question never
# creates or confirms a proposal; it's left for the agent, which calls
# get_booking_status and answers from the real database state instead.
_QUESTION = re.compile(
    r"^\s*(is|was|are|were|has|have|had|does|did|do|can|could|would|will|what|how)\b|\?\s*$", re.IGNORECASE
)


def _looks_like_a_question(message: str) -> bool:
    return bool(_QUESTION.search(message))

_INTERPRET_PROMPT = """An administrator was asked to approve or refuse a parking reservation.
Their reply did not contain a clear "approve" or "refuse" instruction, so decide only between
these two options:
- "refused": the reply clearly declines, rejects or says no to the booking, even if phrased
  unusually (for example "not enough space", "can't do it", "deny this one").
- "unclear": the reply does not clearly refuse either - including if it sounds like an
  approval, is a question, or is off-topic. Never guess an approval; when in doubt, say
  "unclear" so a human can be asked to say APPROVE or REFUSE explicitly.

Extract a short reason if the reply gives one, otherwise leave it empty.

Administrator's reply: {reply}"""


class AmbiguousReplyError(Exception):
    """The reply could not be confidently read as approve or refuse."""

    def __init__(self, reply: str):
        super().__init__(f"Could not tell whether this reply approves or refuses the booking: {reply!r}")
        self.reply = reply


@dataclass(frozen=True)
class InterpretedReply:
    decision: Literal["approved", "refused"]
    reason: str | None


class _ReplyInterpretation(BaseModel):
    """The LLM's conclusion about an unclear reply. 'approved' is deliberately not a
    valid value: once a reply wasn't a clear deterministic approval, the LLM must
    never be the thing that approves it."""

    decision: Literal["refused", "unclear"]
    reason: str | None = None


def render_request(request: ApprovalRequest) -> str:
    """A plain-text summary of the booking, for the administrator to read."""
    lines = [
        f"Booking {request.booking_id} - {request.first_name} {request.last_name}",
        f"Period: {request.start:%Y-%m-%d %H:%M} to {request.end:%Y-%m-%d %H:%M}",
        f"Vehicles ({len(request.vehicles)}):",
    ]
    lines += [f"  - {v.car_number} ({v.vehicle_type or 'unknown type'}) - pass {v.pass_id}" for v in request.vehicles]
    lines.append("Reply with APPROVE or REFUSE (optionally followed by a reason).")
    return "\n".join(lines)


def _extract_reason(reply: str, matched_word_end: int) -> str | None:
    remainder = reply[matched_word_end:].strip(" ,.-:;\n\t")
    return remainder or None


def _check_keywords(reply: str) -> InterpretedReply | None:
    """A clear, unambiguous decision, or None if the LLM should decide instead."""
    if _NEGATED_APPROVAL.search(reply):
        return None
    approve_match = _APPROVE.search(reply)
    refuse_match = _REFUSE.search(reply)
    if approve_match and not refuse_match:
        return InterpretedReply("approved", _extract_reason(reply, approve_match.end()))
    if refuse_match and not approve_match:
        return InterpretedReply("refused", _extract_reason(reply, refuse_match.end()))
    return None  # both matched, or neither: genuinely ambiguous


def _interpret_unclear_reply(reply: str) -> InterpretedReply:
    model = get_chat_model().with_structured_output(_ReplyInterpretation)
    result = model.invoke(_INTERPRET_PROMPT.format(reply=reply))
    if result.decision == "unclear":
        raise AmbiguousReplyError(reply)
    return InterpretedReply("refused", result.reason)


def interpret_reply(reply: str) -> InterpretedReply:
    """Decide approve/refuse from the administrator's free-text reply.

    Raises AmbiguousReplyError if neither the keyword check nor the LLM can tell -
    the caller should ask the administrator to reply again, never guess.
    """
    return _check_keywords(reply) or _interpret_unclear_reply(reply)


# --- The one write path -----------------------------------------------------------------------


class DecisionMismatchError(Exception):
    """The administrator's actual latest message doesn't support the decision a tool
    call claims - the write is refused rather than trusting the claim."""

    def __init__(self, claimed: str, actual: str, admin_message: str):
        super().__init__(f"message reads as {actual!r}, not {claimed!r}: {admin_message!r}")
        self.claimed = claimed
        self.actual = actual


def apply_decision(booking_id: str, decision: Literal["approved", "refused"], reason: str | None) -> AdminDecision:
    """The one path every write goes through - called by interpret_and_apply below,
    and directly by the deterministic confirm step in update_proposal.

    Idempotent: calling this again with the same decision for a booking that already
    has it is treated as success, not a conflict - only a genuinely different second
    decision still raises AlreadyDecidedError.
    """
    try:
        admin_db.record_decision(booking_id, decision, reason)
    except admin_db.AlreadyDecidedError as error:
        if error.status != decision:
            raise
    if decision == "approved":
        from parking_chatbot.recording.client import send_approved_booking

        send_approved_booking(booking_id)  # best-effort; never raises, never blocks the approval
    return AdminDecision(booking_id=booking_id, decision=decision, reason=reason)


def interpret_and_apply(
    booking_id: str,
    admin_message: str,
    expected_decision: Literal["approved", "refused"] | None = None,
    reason_override: str | None = None,
) -> AdminDecision:
    """Derive the decision from the administrator's raw message (interpret_reply),
    optionally requiring it to match `expected_decision`, then write it through
    apply_decision. Used by the REST API and by the record_decision tool below - the
    one difference between them is where the raw message and expected_decision come
    from, never the write logic itself.
    """
    interpreted = interpret_reply(admin_message)
    if expected_decision is not None and interpreted.decision != expected_decision:
        raise DecisionMismatchError(expected_decision, interpreted.decision, admin_message)
    reason = reason_override if reason_override is not None else interpreted.reason
    return apply_decision(booking_id, interpreted.decision, reason)


# --- The deterministic confirmation flow ------------------------------------------------------
#
# Creating, confirming, cancelling or replacing a decision proposal is plain code,
# driven only by the administrator's raw message - never by the agent or an LLM. See
# admin_chat() below for how this runs before the agent is even invoked.


@dataclass(frozen=True)
class _Proposal:
    booking_id: str
    decision: Literal["approved", "refused"]
    reason: str | None


@dataclass(frozen=True)
class ProposalOutcome:
    action: Literal["proposed", "applied", "cancelled", "unclear", "error", "none"]
    booking: ApprovalRequest | None = None
    booking_id: str | None = None
    decision: Literal["approved", "refused"] | None = None
    reason: str | None = None
    message: str | None = None


# One proposal per conversation, keyed by thread_id - not yet written anywhere.
_proposals: dict[str, _Proposal] = {}
# The administrator's actual latest message, keyed by thread_id. Only admin_chat()
# (called from the CLI's own input loop) ever writes this - record_decision reads it,
# an LLM tool call can never supply or overwrite it.
_latest_admin_message: dict[str, str] = {}
# The booking ID(s) currently "in view" for this conversation, keyed by thread_id -
# set whenever list_pending_requests, get_request_details or get_booking_status runs,
# or a proposal is created by naming an ID explicitly. Lets "approve it" resolve
# without repeating the ID: if there's exactly one, it means that one; if there are
# several, the administrator is asked which.
_context_bookings: dict[str, list[str]] = {}


def record_admin_message(thread_id: str, message: str) -> None:
    """The one trusted copy of what the administrator actually typed this turn."""
    _latest_admin_message[thread_id] = message


def set_context_bookings(thread_id: str, booking_ids: list[str]) -> None:
    """Record which booking(s) are currently in view for this conversation."""
    _context_bookings[thread_id] = booking_ids


def _propose(thread_id: str, booking_id: str, decided: InterpretedReply) -> ProposalOutcome:
    """Create (or replace) a proposal for an explicitly-identified booking."""
    booking = admin_db.get_pending_request(booking_id)
    if booking is None:
        return ProposalOutcome(
            action="error",
            message=f"No pending booking found with ID {booking_id} "
            "(it may not exist, already be decided, or have expired).",
        )
    _proposals[thread_id] = _Proposal(booking_id, decided.decision, decided.reason)
    set_context_bookings(thread_id, [booking_id])
    return ProposalOutcome(action="proposed", booking=booking, decision=decided.decision, reason=decided.reason)


def update_proposal(thread_id: str, message: str) -> ProposalOutcome:
    """Run the administrator's raw message through the deterministic confirmation
    flow. Returns action="none" if nothing decision-related was found, so the caller
    can fall back to the agent (for browsing and open-ended conversation).
    """
    if _looks_like_a_question(message):
        # "is PK-X approved?" contains "approved", which matches _APPROVE - but this
        # is a status question, not a decision. Always defer to the agent, which
        # calls get_booking_status and answers from the real database state. A
        # proposal already waiting for yes/no is left untouched, not auto-confirmed.
        return ProposalOutcome(action="none")

    match = _BOOKING_ID_SEARCH.search(message)
    if match:
        booking_id = match.group(0).upper()
        # Strip the ID out before reading the decision/reason, so the ID itself is
        # never mistaken for a reason ("approve PK-ABCDEF" must not yield reason="PK-ABCDEF").
        remainder = (message[: match.start()] + message[match.end() :]).strip()
        if not (_APPROVE.search(remainder) or _REFUSE.search(remainder)):
            return ProposalOutcome(action="none")  # a plain lookup, e.g. "tell me about PK-ABCDEF"

        decided = _check_keywords(remainder)
        if decided is None:
            return ProposalOutcome(
                action="unclear",
                message=f"Couldn't tell whether you want to approve or refuse {booking_id}. "
                "Please say APPROVE or REFUSE clearly.",
            )
        return _propose(thread_id, booking_id, decided)

    proposal = _proposals.get(thread_id)
    if proposal is None:
        # No explicit ID and nothing proposed yet - "approve it" / "refuse, dates
        # clash": a decision word with no ID resolves against whichever booking(s)
        # are currently in view, instead of being rejected for not naming an ID.
        if not (_APPROVE.search(message) or _REFUSE.search(message)):
            return ProposalOutcome(action="none")  # plain browsing or chit-chat

        decided = _check_keywords(message)
        if decided is None:
            return ProposalOutcome(
                action="unclear",
                message="Couldn't tell whether you want to approve or refuse. Please say APPROVE or REFUSE clearly.",
            )

        in_view = _context_bookings.get(thread_id, [])
        if not in_view:
            return ProposalOutcome(action="unclear", message="Which booking do you mean? Please give a booking ID.")
        if len(in_view) > 1:
            return ProposalOutcome(
                action="unclear",
                message=f"Several bookings are in view ({', '.join(in_view)}) - which one do you mean? "
                "Please give a booking ID.",
            )
        return _propose(thread_id, in_view[0], decided)

    is_confirm = bool(_APPROVE.search(message)) and not _NEGATED_APPROVAL.search(message) and not _REFUSE.search(message)
    is_cancel = bool(_REFUSE.search(message)) and not _APPROVE.search(message)

    if is_confirm:
        try:
            result = apply_decision(proposal.booking_id, proposal.decision, proposal.reason)
        except admin_db.AdminBookingError as error:
            _proposals.pop(thread_id, None)
            return ProposalOutcome(action="error", message=str(error))
        _proposals.pop(thread_id, None)
        return ProposalOutcome(action="applied", booking_id=result.booking_id, decision=result.decision, reason=result.reason)

    if is_cancel:
        _proposals.pop(thread_id, None)
        return ProposalOutcome(action="cancelled", booking_id=proposal.booking_id)

    return ProposalOutcome(action="unclear", message="Please reply yes/confirm or no/cancel.")


# --- The agent's own tools ----------------------------------------------------------------------
#
# These exist for browsing (listing, showing details) plus a defense-in-depth
# decision tool - the normal confirm/cancel flow above never calls them.

# Booking IDs the administrator has been shown the full details of, per thread - a
# code-enforced confirmation gate: record_decision refuses to run for any booking it
# hasn't been shown for in this conversation.
_detail_shown: dict[str, set[str]] = {}


def _thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


@tool
def list_pending_requests(config: RunnableConfig) -> str:
    """List every booking currently awaiting a decision, with a short summary of each."""
    tid = _thread_id(config)
    pending = admin_db.list_pending_bookings()
    set_context_bookings(tid, [r.booking_id for r in pending])
    return json.dumps(
        {"count": len(pending), "pending": [{"booking_id": r.booking_id, "summary": render_request(r)} for r in pending]}
    )


@tool
def get_request_details(config: RunnableConfig, booking_id: str) -> str:
    """Show the full details of one pending booking (booker, period, every vehicle) so
    it can be read back to the administrator. Call this before discussing or deciding
    on any specific PENDING booking. For a booking that might already be decided or
    expired, call get_booking_status instead."""
    tid = _thread_id(config)
    booking_id = booking_id.strip().upper()
    if not _BOOKING_ID_FULLMATCH.match(booking_id):
        return json.dumps({"error": f"'{booking_id}' is not a valid booking ID. It should look like PK-XXXXXX."})

    request = admin_db.get_pending_request(booking_id)
    if request is None:
        return json.dumps(
            {"error": f"No pending booking found with ID {booking_id} (it may not exist, already be decided, or have expired)."}
        )

    _detail_shown.setdefault(tid, set()).add(booking_id)
    set_context_bookings(tid, [booking_id])
    return json.dumps(
        {
            "details": request.model_dump(mode="json"),
            "summary": render_request(request),
            "next_step": "Read this back to the administrator and ask them to explicitly reply APPROVE or "
            "REFUSE (with an optional reason) before recording anything.",
        }
    )


@tool
def get_booking_status(config: RunnableConfig, booking_id: str) -> str:
    """Look up the REAL, current status of any booking - pending, approved, refused
    or expired - straight from the database. ALWAYS call this when the administrator
    asks about a booking's status, or says they already decided on one: never answer
    such a question from conversation memory, and never ask them to repeat a decision
    without checking here first. Returns status, decided_at, decision_reason and the
    booking's details (you may see everything, there is no privacy restriction here)."""
    tid = _thread_id(config)
    booking_id = booking_id.strip().upper()
    if not _BOOKING_ID_FULLMATCH.match(booking_id):
        return json.dumps({"error": f"'{booking_id}' is not a valid booking ID. It should look like PK-XXXXXX."})

    info = admin_db.get_booking_for_admin(booking_id)
    if info is None:
        return json.dumps({"error": f"No booking found with ID {booking_id}."})

    if info["status"] == "pending":
        _detail_shown.setdefault(tid, set()).add(booking_id)
        set_context_bookings(tid, [booking_id])
    return json.dumps(info)


@tool
def record_decision(config: RunnableConfig, booking_id: str, decision: Literal["approved", "refused"], reason: str | None = None) -> str:
    """Record the administrator's decision on a booking. Only call this after
    get_request_details or get_booking_status has been shown for this exact
    booking_id in this conversation, and the administrator has explicitly confirmed
    with a clear APPROVE or REFUSE in their own latest message - never guess or infer
    a decision from earlier context. This independently re-checks the administrator's
    actual latest message before writing anything, and refuses if that message
    doesn't clearly support `decision`."""
    tid = _thread_id(config)
    booking_id = booking_id.strip().upper()
    if booking_id not in _detail_shown.get(tid, set()):
        return json.dumps(
            {"error": f"Call get_request_details or get_booking_status for {booking_id} first, and get the "
             "administrator's explicit confirmation, before recording a decision."}
        )

    admin_message = _latest_admin_message.get(tid)
    if not admin_message:
        return json.dumps({"error": "No administrator message is on record for this turn; cannot verify the decision."})

    try:
        result = interpret_and_apply(booking_id, admin_message, expected_decision=decision, reason_override=reason)
    except AmbiguousReplyError:
        return json.dumps({"error": "Your latest message doesn't clearly say APPROVE or REFUSE. Please state it plainly."})
    except DecisionMismatchError as error:
        return json.dumps(
            {"error": f"Your latest message reads as {error.actual}, not {decision} as requested. "
             "Nothing was recorded; please confirm clearly."}
        )
    except admin_db.UnknownBookingError as error:
        return json.dumps({"error": str(error)})
    except admin_db.ExpiredBookingError as error:
        return json.dumps({"error": str(error)})
    except admin_db.AlreadyDecidedError as error:
        return json.dumps({"error": str(error)})

    _detail_shown.get(tid, set()).discard(booking_id)
    return json.dumps({"booking_id": result.booking_id, "decision": result.decision, "reason": result.reason})


ADMIN_TOOLS = [list_pending_requests, get_request_details, get_booking_status, record_decision]

ADMIN_AGENT_SYSTEM_PROMPT = """You are the administrator's assistant for CityPark Central parking reservations.

Your job is to help the administrator review bookings and find out their real
status. You are talking to the administrator, not a customer - there is no privacy
restriction on the booking details shown here.

Rules:
1. To see what's waiting, call list_pending_requests.
2. Before discussing or deciding on a specific PENDING booking, call
   get_request_details for its booking ID so the administrator sees the full details.
3. Whenever the administrator asks about a booking's status, or says they already
   approved/refused/decided on something, you MUST call get_booking_status and
   answer strictly from its result - never from what was said earlier in this
   conversation. If it reports the booking is already approved, refused or expired,
   say so plainly (when it was decided, and the reason if refused) and STOP - do not
   ask the administrator to repeat the decision, and never say a decision "hasn't
   been recorded" or is "still pending" without having just checked.
4. Most approve/refuse decisions are handled automatically before you are even asked:
   a message naming a booking with a clear decision ("approve PK-X", "refuse PK-X,
   no space"), "approve it" / "refuse it" when exactly one booking is currently in
   view, or a plain yes/no confirming a decision that was just proposed, are all
   acted on by the system before you see them. You are only asked to help when that
   didn't resolve things - for example several bookings are in view and it's unclear
   which one "it" means, or the wording was genuinely ambiguous.
5. If it's unclear which booking is meant, or whether the administrator means
   approve or refuse, ask them to say which booking (or its ID) and APPROVE or REFUSE
   plainly. Never guess, and never call record_decision based on a guess, an earlier
   message, or anything other than what the administrator just typed.
6. Only call record_decision as a last resort, and only right after the
   administrator's own explicit APPROVE/REFUSE for that exact booking ID. It
   independently re-checks their actual latest message and refuses to write if
   anything is unclear or contradictory - if it returns an error (unknown booking,
   already decided, expired, ambiguous, or a mismatch), relay that specific reason
   plainly and ask the administrator to restate their decision.
7. Never claim a booking was approved or refused unless a tool actually reports that
   outcome. Never invent booking details.
"""


def build_admin_agent():
    """Create the admin agent: LLM + tools + system prompt + conversation memory."""
    return create_agent(
        model=get_chat_model(),
        tools=ADMIN_TOOLS,
        system_prompt=ADMIN_AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def _invoke_admin_agent(agent, admin_message: str, thread_id: str) -> tuple[str, list[str]]:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": admin_message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    tools_used = [m.name for m in messages[last_human + 1 :] if m.type == "tool"]
    return messages[-1].content, tools_used


def admin_chat(agent, admin_message: str, thread_id: str) -> tuple[str, list[str]]:
    """The administrator's entry point. The deterministic confirmation flow runs
    first, driven only by the administrator's own raw message; the agent is invoked
    only when nothing decision-related was found (browsing, open-ended conversation)."""
    record_admin_message(thread_id, admin_message)
    outcome = update_proposal(thread_id, admin_message)

    if outcome.action == "proposed":
        verb = "APPROVE" if outcome.decision == "approved" else "REFUSE"
        reason_note = f" (reason: {outcome.reason})" if outcome.reason else ""
        return (
            f"{render_request(outcome.booking)}\n\n"
            f"You're about to {verb} this booking{reason_note}. "
            "Reply yes/confirm to proceed, or no/cancel to back out.",
            [],
        )
    if outcome.action == "applied":
        return f"Done - booking {outcome.booking_id} is now {outcome.decision.upper()}.", []
    if outcome.action == "cancelled":
        return f"Cancelled. Booking {outcome.booking_id} was not changed.", []
    if outcome.action in ("unclear", "error"):
        return outcome.message, []

    return _invoke_admin_agent(agent, admin_message, thread_id)
