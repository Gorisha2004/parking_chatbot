import json
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from parking_chatbot import config
from parking_chatbot.admin import agent
from parking_chatbot.admin.agent import AmbiguousReplyError, DecisionMismatchError, interpret_reply, render_request
from parking_chatbot.admin.models import ApprovalRequest, RequestVehicle
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.database import get_connection


def _insert_pending_booking(booking_id: str = "PK-AAAAAA", created_at: datetime | None = None) -> None:
    with closing(get_connection()) as conn, conn:
        if created_at is None:
            cursor = conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, escalated_at) "
                "VALUES (?, 'Anna', 'Smith', '2026-11-20T09:00', '2026-11-20T12:00', '2026-11-18T00:00')",
                (booking_id,),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, "
                "escalated_at, created_at) VALUES (?, 'Anna', 'Smith', '2026-11-20T09:00', "
                "'2026-11-20T12:00', '2026-11-18T00:00', ?)",
                (booking_id, created_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
        conn.execute(
            "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
            "VALUES (?, ?, 'AB123CD', 'car')",
            (cursor.lastrowid, f"{booking_id}-01"),
        )


def _new_thread() -> str:
    return str(uuid.uuid4())


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        booking_id="PK-7K3Q9X",
        first_name="Anna",
        last_name="Smith",
        start=datetime(2026, 11, 20, 9, 0),
        end=datetime(2026, 11, 20, 12, 0),
        vehicles=[
            RequestVehicle(pass_id="PK-7K3Q9X-01", car_number="AB123CD", vehicle_type="car"),
            RequestVehicle(pass_id="PK-7K3Q9X-02", car_number="XY987ZT", vehicle_type=None),
        ],
    )


# --- render_request ----------------------------------------------------------------------------


def test_render_request_includes_booking_name_period_and_every_vehicle():
    text = render_request(_request())
    assert "PK-7K3Q9X" in text
    assert "Anna Smith" in text
    assert "AB123CD" in text and "XY987ZT" in text
    assert "PK-7K3Q9X-01" in text and "PK-7K3Q9X-02" in text


def test_render_request_handles_a_vehicle_with_no_recorded_type():
    assert "unknown type" in render_request(_request())


# --- interpret_reply: deterministic keyword path (no LLM call) -----------------------------------


@pytest.mark.parametrize(
    "reply", ["approve", "Approved", "yes", "APPROVE this one", "confirmed, looks good", "ok"]
)
def test_clear_approval_keywords_are_read_deterministically(reply, monkeypatch):
    monkeypatch.setattr(agent, "_interpret_unclear_reply", lambda r: pytest.fail("must not call the LLM"))
    assert interpret_reply(reply).decision == "approved"


@pytest.mark.parametrize("reply", ["refuse", "Rejected", "no", "deny this one", "refused - no space"])
def test_clear_refusal_keywords_are_read_deterministically(reply, monkeypatch):
    monkeypatch.setattr(agent, "_interpret_unclear_reply", lambda r: pytest.fail("must not call the LLM"))
    assert interpret_reply(reply).decision == "refused"


def test_extracts_a_reason_after_the_keyword():
    result = interpret_reply("refuse - not enough space that day")
    assert result.decision == "refused"
    assert result.reason == "not enough space that day"


def test_approval_with_no_trailing_text_has_no_reason():
    assert interpret_reply("approve").reason is None


@pytest.mark.parametrize("reply", ["do not approve", "don't confirm this", "never approve this one"])
def test_negated_approval_is_never_read_as_a_plain_approval(reply, monkeypatch):
    # Falls through to the LLM path, where "approved" cannot be produced - so the net
    # effect is always refused/unclear, never a false approval.
    monkeypatch.setattr(agent, "_interpret_unclear_reply", lambda r: agent.InterpretedReply("refused", None))
    assert interpret_reply(reply).decision == "refused"


# --- interpret_reply: LLM fallback for genuinely unclear text -------------------------------------


def test_unclear_reply_with_no_keywords_falls_back_to_the_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent, "_interpret_unclear_reply", lambda r: calls.append(r) or agent.InterpretedReply("refused", "n/a")
    )
    result = interpret_reply("the dates look a bit odd to me")
    assert calls == ["the dates look a bit odd to me"]
    assert result.decision == "refused"


def test_reply_with_both_approve_and_refuse_words_is_ambiguous_not_guessed(monkeypatch):
    monkeypatch.setattr(agent, "_interpret_unclear_reply", lambda r: agent.InterpretedReply("refused", None))
    # contains both an approve-word ("yes") and a refuse-word ("no") - must not pick one blindly
    assert interpret_reply("yes, no, actually not sure").decision == "refused"


def test_llm_reporting_unclear_raises_ambiguous_reply_error(monkeypatch):
    class _FakeStructured:
        def invoke(self, prompt):
            class _Result:
                decision = "unclear"
                reason = None

            return _Result()

    monkeypatch.setattr(agent, "get_chat_model", lambda: type("M", (), {"with_structured_output": lambda self, s: _FakeStructured()})())
    with pytest.raises(AmbiguousReplyError):
        interpret_reply("hmm, not sure about this one")


def test_llm_structured_output_cannot_express_approved():
    # The schema itself forbids it - this is the hard guarantee, not just prompt wording.
    assert "approved" not in agent._ReplyInterpretation.model_fields["decision"].annotation.__args__


# --- apply_decision / interpret_and_apply: the one write path ------------------------------------


def test_apply_decision_writes_the_decision(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    result = agent.apply_decision("PK-AAAAAA", "approved", "fine")
    assert result.decision == "approved"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"


def test_apply_decision_is_idempotent_for_a_repeated_identical_decision(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    agent.apply_decision("PK-AAAAAA", "approved", "fine")

    result = agent.apply_decision("PK-AAAAAA", "approved", "fine")  # e.g. a retried call

    assert result.decision == "approved"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"


def test_apply_decision_still_rejects_a_genuinely_conflicting_second_decision(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    agent.apply_decision("PK-AAAAAA", "approved", None)

    with pytest.raises(admin_db.AlreadyDecidedError):
        agent.apply_decision("PK-AAAAAA", "refused", None)
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"


def test_interpret_and_apply_derives_the_decision_from_the_message(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    result = agent.interpret_and_apply("PK-AAAAAA", "approve, looks fine")
    assert (result.decision, result.reason) == ("approved", "looks fine")


def test_interpret_and_apply_raises_when_the_message_contradicts_expected_decision(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    with pytest.raises(DecisionMismatchError):
        agent.interpret_and_apply("PK-AAAAAA", "refuse this one", expected_decision="approved")
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"


def test_interpret_and_apply_raises_on_an_ambiguous_message(sql_db, monkeypatch):
    _insert_pending_booking("PK-AAAAAA")
    monkeypatch.setattr(agent, "_interpret_unclear_reply", lambda r: (_ for _ in ()).throw(AmbiguousReplyError(r)))
    with pytest.raises(AmbiguousReplyError):
        agent.interpret_and_apply("PK-AAAAAA", "not sure about this one")
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"


def test_interpret_and_apply_prefers_an_explicit_reason_override(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    result = agent.interpret_and_apply("PK-AAAAAA", "approve", reason_override="custom reason")
    assert result.reason == "custom reason"


# --- update_proposal: the deterministic confirmation flow ----------------------------------------


def test_update_proposal_does_nothing_for_a_plain_lookup(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    outcome = agent.update_proposal(_new_thread(), "tell me about PK-AAAAAA")
    assert outcome.action == "none"


@pytest.mark.parametrize(
    "message",
    [
        "is PK-AAAAAA approved?",
        "has PK-AAAAAA been approved yet?",
        "was PK-AAAAAA refused?",
        "did you approve PK-AAAAAA",
    ],
)
def test_update_proposal_never_treats_a_status_question_as_a_decision(sql_db, message):
    """Root cause of the bug report: 'approved'/'refused' match the decision
    keywords, so a status QUESTION must be recognised and left for the agent
    (get_booking_status) instead of silently creating or confirming a proposal."""
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()

    outcome = agent.update_proposal(tid, message)

    assert outcome.action == "none"
    assert tid not in agent._proposals


def test_update_proposal_leaves_a_pending_proposal_untouched_when_asked_an_unrelated_question(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")

    outcome = agent.update_proposal(tid, "is it approved yet?")

    assert outcome.action == "none"  # left for the agent to answer
    assert agent._proposals[tid].booking_id == "PK-AAAAAA"  # proposal still intact, not auto-confirmed
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"  # and nothing was written


def test_update_proposal_creates_a_proposal_from_a_clear_decision(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    outcome = agent.update_proposal(_new_thread(), "approve PK-AAAAAA, looks fine")
    assert (outcome.action, outcome.decision, outcome.reason) == ("proposed", "approved", "looks fine")
    assert outcome.booking.booking_id == "PK-AAAAAA"


def test_update_proposal_reports_an_unknown_booking_without_creating_a_proposal(sql_db):
    tid = _new_thread()
    outcome = agent.update_proposal(tid, "approve PK-ZZZZZZ")
    assert outcome.action == "error"
    assert tid not in agent._proposals


def test_update_proposal_is_unclear_for_an_ambiguous_decision_naming_a_booking(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    outcome = agent.update_proposal(tid, "PK-AAAAAA, not sure if I should approve or refuse")
    assert outcome.action == "unclear"
    assert tid not in agent._proposals


@pytest.mark.parametrize("reply", ["don't approve PK-AAAAAA", "never approve PK-AAAAAA"])
def test_update_proposal_never_proposes_an_approval_from_negated_text(sql_db, reply):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    outcome = agent.update_proposal(tid, reply)
    assert outcome.action == "unclear"
    assert tid not in agent._proposals


def test_update_proposal_applies_a_bare_yes_confirmation(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA, fine")

    outcome = agent.update_proposal(tid, "yes")

    assert (outcome.action, outcome.booking_id, outcome.decision) == ("applied", "PK-AAAAAA", "approved")
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"
    assert admin_db.get_booking("PK-AAAAAA")["decision_reason"] == "fine"
    assert tid not in agent._proposals


def test_update_proposal_confirming_a_refusal_proposal_with_yes_still_refuses(sql_db):
    # Regression: interpreting "yes" on its own always reads as approved, so applying
    # a stored REFUSAL proposal must use the stored decision, not re-derive it from "yes".
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "refuse PK-AAAAAA, no space")

    outcome = agent.update_proposal(tid, "yes")

    assert outcome.decision == "refused"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "refused"


def test_update_proposal_cancels_a_proposal_on_a_bare_no(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")

    outcome = agent.update_proposal(tid, "no")

    assert outcome.action == "cancelled"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"
    assert tid not in agent._proposals


def test_update_proposal_is_unclear_when_neither_confirm_nor_cancel_is_clear(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")

    outcome = agent.update_proposal(tid, "maybe")

    assert outcome.action == "unclear"
    assert tid in agent._proposals  # left intact for a retry


def test_update_proposal_replaces_an_existing_proposal_with_a_new_one(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    _insert_pending_booking("PK-BBBBBB")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")

    outcome = agent.update_proposal(tid, "refuse PK-BBBBBB, no space")

    assert (outcome.booking.booking_id, outcome.decision) == ("PK-BBBBBB", "refused")
    assert agent._proposals[tid].booking_id == "PK-BBBBBB"


def test_update_proposal_reports_an_error_if_the_booking_was_decided_elsewhere_before_confirmation(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")
    admin_db.record_decision("PK-AAAAAA", "refused", "decided elsewhere")  # e.g. via the REST API

    outcome = agent.update_proposal(tid, "yes")

    assert outcome.action == "error"
    assert tid not in agent._proposals


# --- update_proposal: "approve it" resolves against whatever's in context ------------------------


def test_update_proposal_resolves_approve_it_when_exactly_one_booking_is_in_context(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.set_context_bookings(tid, ["PK-AAAAAA"])  # e.g. just shown via get_request_details

    outcome = agent.update_proposal(tid, "approve it")

    assert (outcome.action, outcome.booking.booking_id, outcome.decision) == ("proposed", "PK-AAAAAA", "approved")


def test_update_proposal_resolves_a_bare_decision_with_a_reason_against_the_one_in_context(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.set_context_bookings(tid, ["PK-AAAAAA"])

    outcome = agent.update_proposal(tid, "refuse, dates clash")

    assert (outcome.booking.booking_id, outcome.decision, outcome.reason) == ("PK-AAAAAA", "refused", "dates clash")


def test_update_proposal_asks_which_booking_when_several_are_in_context(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    _insert_pending_booking("PK-BBBBBB")
    tid = _new_thread()
    agent.set_context_bookings(tid, ["PK-AAAAAA", "PK-BBBBBB"])

    outcome = agent.update_proposal(tid, "approve it")

    assert outcome.action == "unclear"
    assert "PK-AAAAAA" in outcome.message and "PK-BBBBBB" in outcome.message
    assert tid not in agent._proposals


def test_update_proposal_asks_for_an_id_when_nothing_is_in_context(sql_db):
    tid = _new_thread()
    outcome = agent.update_proposal(tid, "approve it")
    assert outcome.action == "unclear"
    assert "booking ID" in outcome.message


def test_update_proposal_maybe_never_approves_with_one_booking_in_context(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.set_context_bookings(tid, ["PK-AAAAAA"])

    outcome = agent.update_proposal(tid, "maybe")

    assert outcome.action == "none"  # no decision language at all - left for the agent
    assert tid not in agent._proposals


def test_update_proposal_maybe_never_approves_with_a_proposal_pending(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.update_proposal(tid, "approve PK-AAAAAA")

    outcome = agent.update_proposal(tid, "maybe")

    assert outcome.action == "unclear"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"


def test_list_pending_requests_tool_sets_the_context_for_approve_it(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    tid = _new_thread()
    agent.list_pending_requests.invoke({}, _cfg(tid))

    outcome = agent.update_proposal(tid, "approve it")

    assert outcome.booking.booking_id == "PK-AAAAAA"


# --- the agent's tools -----------------------------------------------------------------------------


def _cfg(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or _new_thread()}}


def test_list_pending_requests_tool_returns_a_summary_per_booking(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    result = agent.list_pending_requests.invoke({}, _cfg())
    data = json.loads(result)
    assert data["count"] == 1
    assert data["pending"][0]["booking_id"] == "PK-AAAAAA"


def test_get_request_details_tool_returns_full_details_and_marks_it_shown(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    cfg = _cfg()

    result = json.loads(agent.get_request_details.invoke({"booking_id": "PK-AAAAAA"}, cfg))

    assert result["details"]["booking_id"] == "PK-AAAAAA"
    assert "PK-AAAAAA" in agent._detail_shown[cfg["configurable"]["thread_id"]]


def test_get_request_details_tool_reports_an_unknown_or_non_pending_booking(sql_db):
    result = json.loads(agent.get_request_details.invoke({"booking_id": "PK-ZZZZZZ"}, _cfg()))
    assert "error" in result


def test_get_request_details_tool_rejects_an_invalid_id_format(sql_db):
    result = json.loads(agent.get_request_details.invoke({"booking_id": "not-an-id"}, _cfg()))
    assert "not a valid booking ID" in result["error"]


# --- get_booking_status: the real status, for any booking, never from memory ---------------------


def test_get_booking_status_tool_reports_a_decided_booking(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    admin_db.record_decision("PK-AAAAAA", "refused", "vehicle number doesn't look right")

    result = json.loads(agent.get_booking_status.invoke({"booking_id": "PK-AAAAAA"}, _cfg()))

    assert result["status"] == "refused"
    assert result["decision_reason"] == "vehicle number doesn't look right"
    assert result["decided_at"] is not None


def test_get_booking_status_tool_reports_an_expired_booking(sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    overdue = datetime.now(timezone.utc) - timedelta(minutes=61)
    _insert_pending_booking("PK-AAAAAA", created_at=overdue)

    result = json.loads(agent.get_booking_status.invoke({"booking_id": "PK-AAAAAA"}, _cfg()))

    assert result["status"] == "expired"


def test_get_booking_status_tool_reports_a_pending_booking_and_its_details(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    result = json.loads(agent.get_booking_status.invoke({"booking_id": "PK-AAAAAA"}, _cfg()))
    assert result["status"] == "pending"
    assert (result["first_name"], result["last_name"]) == ("Anna", "Smith")


def test_get_booking_status_tool_reports_an_unknown_id(sql_db):
    result = json.loads(agent.get_booking_status.invoke({"booking_id": "PK-ZZZZZZ"}, _cfg()))
    assert "error" in result


def test_get_booking_status_sees_a_decision_made_in_a_completely_different_session(sql_db):
    """Regression for the bug report: the status must come from the database, not
    from this (or any other) conversation's own memory - a fresh thread that never
    discussed the booking before must still see a decision made elsewhere."""
    _insert_pending_booking("PK-AAAAAA")
    admin_db.record_decision("PK-AAAAAA", "approved", None)  # e.g. decided in an earlier admin session

    fresh_thread_result = json.loads(agent.get_booking_status.invoke({"booking_id": "PK-AAAAAA"}, _cfg(_new_thread())))

    assert fresh_thread_result["status"] == "approved"


def test_get_booking_status_tool_lets_record_decision_run_for_a_pending_booking(sql_db):
    """Calling get_booking_status (not just get_request_details) also satisfies the
    confirmation gate, since it shows the same details."""
    _insert_pending_booking("PK-AAAAAA")
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent.get_booking_status.invoke({"booking_id": "PK-AAAAAA"}, cfg)
    agent.record_admin_message(tid, "approve")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "approved"}, cfg))

    assert result["decision"] == "approved"


def test_record_decision_tool_refuses_without_get_request_details_first(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    cfg = _cfg()
    agent.record_admin_message(cfg["configurable"]["thread_id"], "approve PK-AAAAAA")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "approved"}, cfg))

    assert "get_request_details" in result["error"]
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"


def test_record_decision_tool_ignores_a_forged_admin_message_and_uses_the_trusted_one(sql_db):
    """The tool's schema has no admin_message argument - even if the LLM's reasoning
    led it to call this with decision="approved", only the administrator's real last
    message (set via record_admin_message, not by the tool call) is trusted."""
    _insert_pending_booking("PK-AAAAAA")
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent.get_request_details.invoke({"booking_id": "PK-AAAAAA"}, cfg)
    agent.record_admin_message(tid, "refuse this one, not enough space")  # the administrator's real words

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "approved"}, cfg))

    assert "error" in result
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"  # nothing was written


def test_record_decision_tool_succeeds_when_the_real_message_matches(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent.get_request_details.invoke({"booking_id": "PK-AAAAAA"}, cfg)
    agent.record_admin_message(tid, "approve, looks fine")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "approved"}, cfg))

    assert (result["booking_id"], result["decision"]) == ("PK-AAAAAA", "approved")
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"


def test_record_decision_tool_reports_an_already_decided_booking_with_the_decision_and_when(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    admin_db.record_decision("PK-AAAAAA", "approved", "looks fine")  # e.g. decided via the REST API
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent._detail_shown.setdefault(tid, set()).add("PK-AAAAAA")  # simulate having been shown it
    agent.record_admin_message(tid, "refuse")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "refused"}, cfg))

    assert "error" in result
    assert "approved" in result["error"] and "looks fine" in result["error"]
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"  # unchanged


def test_record_decision_tool_reports_an_expired_booking_distinctly(sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    overdue = datetime.now(timezone.utc) - timedelta(minutes=61)
    _insert_pending_booking("PK-AAAAAA", created_at=overdue)
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent._detail_shown.setdefault(tid, set()).add("PK-AAAAAA")
    agent.record_admin_message(tid, "approve")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-AAAAAA", "decision": "approved"}, cfg))

    assert "expired" in result["error"]
    assert "already" not in result["error"]  # distinct from the "already decided" wording


def test_record_decision_tool_reports_an_unknown_booking(sql_db):
    cfg = _cfg()
    tid = cfg["configurable"]["thread_id"]
    agent._detail_shown.setdefault(tid, set()).add("PK-ZZZZZZ")  # simulate having "shown" it
    agent.record_admin_message(tid, "approve")

    result = json.loads(agent.record_decision.invoke({"booking_id": "PK-ZZZZZZ", "decision": "approved"}, cfg))

    assert "error" in result


# --- build_admin_agent / admin_chat ----------------------------------------------------------------


def test_build_admin_agent_wires_model_tools_prompt_and_checkpointer(monkeypatch):
    captured = {}

    def fake_create_agent(*, model, tools, system_prompt, checkpointer):
        captured.update(model=model, tools=tools, system_prompt=system_prompt, checkpointer=checkpointer)
        return "built-admin-agent"

    monkeypatch.setattr(agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent, "get_chat_model", lambda: "fake-model")

    result = agent.build_admin_agent()

    assert result == "built-admin-agent"
    assert captured["model"] == "fake-model"
    assert captured["tools"] == agent.ADMIN_TOOLS
    assert captured["system_prompt"] == agent.ADMIN_AGENT_SYSTEM_PROMPT
    assert captured["checkpointer"] is not None


@dataclass
class _FakeMessage:
    type: str
    content: str
    name: str | None = None


@dataclass
class _FakeAdminAgent:
    reply: str
    tool_names: list[str] = field(default_factory=list)
    invoked: bool = False

    def invoke(self, payload, config):
        self.invoked = True
        messages = [_FakeMessage("human", payload["messages"][0]["content"])]
        messages += [_FakeMessage("tool", "{}", name=name) for name in self.tool_names]
        messages.append(_FakeMessage("ai", self.reply))
        return {"messages": messages}


def test_admin_chat_handles_a_proposal_deterministically_without_calling_the_agent(sql_db):
    _insert_pending_booking("PK-AAAAAA")
    fake_agent = _FakeAdminAgent(reply="should not be reached")

    answer, tools_used = agent.admin_chat(fake_agent, "approve PK-AAAAAA, fine", _new_thread())

    assert "You're about to APPROVE this booking (reason: fine)" in answer
    assert fake_agent.invoked is False
    assert tools_used == []


def test_admin_chat_falls_back_to_the_agent_for_plain_conversation():
    fake_agent = _FakeAdminAgent(reply="here's the list", tool_names=["list_pending_requests"])

    answer, tools_used = agent.admin_chat(fake_agent, "what's pending?", _new_thread())

    assert fake_agent.invoked is True
    assert answer == "here's the list"
    assert tools_used == ["list_pending_requests"]
