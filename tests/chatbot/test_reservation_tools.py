import inspect
import json
import re
import uuid
from contextlib import closing
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from parking_chatbot import config
from parking_chatbot.admin.channel import FakeAdminChannel
from parking_chatbot.chatbot import reservation_tools as rt
from parking_chatbot.chatbot.tools import ALL_TOOLS
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.database import get_connection

BOOKING_ID = re.compile(r"^PK-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$")


def _new_cfg():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


@pytest.fixture
def cfg():
    """A fresh conversation per test, plus cleanup of the module-level draft state."""
    run_config = _new_cfg()
    yield run_config
    rt._clear(run_config["configurable"]["thread_id"])


@pytest.fixture
def capacity(sql_db):
    def _set(total: int) -> None:
        with closing(get_connection()) as conn, conn:
            conn.execute("DELETE FROM availability")
            conn.execute(
                "INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, ?, ?)", (total, total)
            )

    _set(100)
    return _set


def _future_times():
    start = datetime.now() + timedelta(days=1)
    end = start + timedelta(hours=2)
    return start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")


def _call(tool, args, run_config):
    return json.loads(tool.invoke(args, run_config))


def _vehicles(*plates, vehicle_type="car"):
    return {"vehicles": [{"car_number": plate, "vehicle_type": vehicle_type} for plate in plates]}


def _fill(run_config, plates=("AB123CD",)):
    start, end = _future_times()
    _call(rt.update_reservation_details, {"first_name": "anna", "last_name": "smith", "start": start, "end": end}, run_config)
    return _call(rt.add_vehicles, _vehicles(*plates), run_config)


# --- Booker and period ---------------------------------------------------------------------------


def test_update_reservation_details_saves_valid_fields(cfg):
    result = _call(rt.update_reservation_details, {"first_name": "anna"}, cfg)
    assert result["saved"] == ["first_name"]
    assert result["current_details"]["first_name"] == "Anna"
    assert "vehicles" in result["still_missing"]


def test_booking_two_years_ahead_is_rejected_and_nothing_is_saved(cfg):
    result = _call(rt.update_reservation_details, {"start": "2028-06-01T09:00", "end": "2028-06-01T12:00"}, cfg)
    assert result["saved"] == []
    assert set(result["rejected"]) == {"start", "end"}
    assert "in advance" in result["rejected"]["start"]


# --- Vehicles ------------------------------------------------------------------------------------


def test_add_vehicles_adds_a_whole_group_in_one_call(cfg):
    result = _call(rt.add_vehicles, _vehicles("AAA111", "BBB222", "CCC333"), cfg)
    assert result["added"] == ["AAA111", "BBB222", "CCC333"]
    assert result["vehicle_count"] == 3
    assert result["max_vehicles"] == config.MAX_VEHICLES_PER_BOOKING


def test_add_vehicles_rejects_invalid_ones_and_keeps_the_valid_ones(cfg):
    args = {"vehicles": [{"car_number": "AAA111", "vehicle_type": "car"}, {"car_number": "!!", "vehicle_type": "car"},
                         {"car_number": "BBB222", "vehicle_type": "truck"}]}
    result = _call(rt.add_vehicles, args, cfg)
    assert result["added"] == ["AAA111"]
    assert set(result["rejected"]) == {"!!", "BBB222"}
    assert "three_wheeler" in result["rejected"]["BBB222"]


def test_adding_a_plate_again_updates_its_vehicle_type(cfg):
    _call(rt.add_vehicles, _vehicles("AAA111"), cfg)
    result = _call(rt.add_vehicles, _vehicles("aaa-111", vehicle_type="van"), cfg)
    assert result["vehicle_count"] == 1
    assert result["current_details"]["vehicles"] == [{"car_number": "AAA111", "vehicle_type": "van"}]


def test_add_vehicles_stops_at_the_configured_group_size(cfg, monkeypatch):
    monkeypatch.setattr(config, "MAX_VEHICLES_PER_BOOKING", 2)
    result = _call(rt.add_vehicles, _vehicles("AAA111", "BBB222", "CCC333"), cfg)
    assert result["added"] == ["AAA111", "BBB222"]
    assert "at most 2 vehicles" in result["rejected"]["CCC333"]


def test_remove_vehicle_takes_it_out_of_the_booking(cfg):
    _call(rt.add_vehicles, _vehicles("AAA111", "BBB222"), cfg)
    result = _call(rt.remove_vehicle, {"car_number": "aaa 111"}, cfg)
    assert result["removed"] is True
    assert [v["car_number"] for v in result["current_details"]["vehicles"]] == ["BBB222"]


def test_remove_vehicle_reports_an_unknown_plate(cfg):
    result = _call(rt.remove_vehicle, {"car_number": "NOPE999"}, cfg)
    assert result["removed"] is False


# --- Summary, capacity and submission ------------------------------------------------------------


def test_reservation_is_incomplete_without_vehicles(cfg, capacity):
    start, end = _future_times()
    result = _call(rt.update_reservation_details, {"first_name": "anna", "last_name": "smith", "start": start, "end": end}, cfg)
    assert result["still_missing"] == ["vehicles"]
    assert "error" in _call(rt.show_reservation_summary, {}, cfg)


def test_submit_reservation_fails_when_incomplete(cfg):
    assert "error" in _call(rt.submit_reservation, {}, cfg)


def test_submit_reservation_fails_without_summary_confirmation(cfg, capacity):
    _fill(cfg)
    result = _call(rt.submit_reservation, {}, cfg)
    assert "show_reservation_summary" in result["error"]


def test_summary_refuses_a_group_larger_than_the_free_spaces(cfg, capacity):
    capacity(2)
    _fill(cfg, plates=("AAA111", "BBB222", "CCC333"))

    summary = _call(rt.show_reservation_summary, {}, cfg)

    assert summary["error"] == "Not enough free spaces for this period"
    assert (summary["free_spaces"], summary["vehicles_in_booking"]) == (2, 3)
    assert "show_reservation_summary" in _call(rt.submit_reservation, {}, cfg)["error"]


def test_full_group_booking_returns_booking_id_and_one_pass_per_vehicle(cfg, capacity):
    _fill(cfg, plates=("AAA111", "BBB222", "CCC333"))
    summary = _call(rt.show_reservation_summary, {}, cfg)
    assert summary["vehicle_count"] == 3

    result = _call(rt.submit_reservation, {}, cfg)

    assert result["status"] == "pending"
    assert BOOKING_ID.match(result["booking_id"])
    assert [(p["pass_id"], p["car_number"]) for p in result["passes"]] == [
        (f"{result['booking_id']}-01", "AAA111"),
        (f"{result['booking_id']}-02", "BBB222"),
        (f"{result['booking_id']}-03", "CCC333"),
    ]
    assert _call(rt.update_reservation_details, {}, cfg)["still_missing"] == rt.REQUIRED_FIELDS


def test_submit_reservation_escalates_through_the_real_admin_channel(cfg, capacity):
    """No fake here: proves the real DbAdminChannel is wired in, not only tested via a fake."""
    _fill(cfg, plates=("AAA111",))
    _call(rt.show_reservation_summary, {}, cfg)
    result = _call(rt.submit_reservation, {}, cfg)
    [pending] = admin_db.list_pending_bookings()
    assert pending.booking_id == result["booking_id"]


def test_submit_reservation_does_not_depend_on_the_admin_server_being_up(cfg, capacity):
    """Escalation must be a local write, never a network call - so submitting a
    reservation can never fail just because the admin server isn't running."""
    _fill(cfg, plates=("AAA111",))
    _call(rt.show_reservation_summary, {}, cfg)
    with patch("socket.socket.connect", side_effect=AssertionError("must not open a network connection")):
        result = _call(rt.submit_reservation, {}, cfg)
    assert result["status"] == "pending"


def test_submit_reservation_sends_the_request_through_whichever_channel_is_injected(cfg, capacity, monkeypatch):
    fake = FakeAdminChannel()
    monkeypatch.setattr(rt, "get_admin_channel", lambda: fake)

    _fill(cfg, plates=("AAA111", "BBB222"))
    _call(rt.show_reservation_summary, {}, cfg)
    result = _call(rt.submit_reservation, {}, cfg)

    [sent] = fake.sent
    assert sent.booking_id == result["booking_id"]
    assert (sent.first_name, sent.last_name) == ("Anna", "Smith")
    assert [v.car_number for v in sent.vehicles] == ["AAA111", "BBB222"]
    # a fake channel never calls mark_escalated, so the admin API would not see it:
    assert admin_db.list_pending_bookings() == []


def test_submit_rechecks_capacity_taken_after_the_summary(cfg, capacity):
    capacity(3)
    _fill(cfg, plates=("AAA111", "BBB222"))
    assert "details" in _call(rt.show_reservation_summary, {}, cfg)

    other = _new_cfg()
    _fill(other, plates=("XXX111", "YYY222"))
    _call(rt.show_reservation_summary, {}, other)
    assert "booking_id" in _call(rt.submit_reservation, {}, other)
    rt._clear(other["configurable"]["thread_id"])

    result = _call(rt.submit_reservation, {}, cfg)
    assert result["error"] == "Not enough free spaces for this period"
    assert "show_reservation_summary" in _call(rt.submit_reservation, {}, cfg)["error"]


def test_any_change_after_the_summary_needs_a_new_confirmation(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    _call(rt.add_vehicles, _vehicles("NEW999"), cfg)
    assert "show_reservation_summary" in _call(rt.submit_reservation, {}, cfg)["error"]


def test_cancel_reservation_clears_the_draft(cfg):
    _fill(cfg)
    rt.cancel_reservation.invoke({}, cfg)
    assert _call(rt.update_reservation_details, {}, cfg)["still_missing"] == rt.REQUIRED_FIELDS


def test_cancel_reservation_reports_a_draft_was_discarded(cfg):
    _fill(cfg)
    result = _call(rt.cancel_reservation, {}, cfg)
    assert result["status"] == "draft discarded"


def test_cancel_reservation_is_honest_when_there_is_nothing_to_discard(cfg):
    """Regression test: calling this with no draft in progress (for example because
    the user actually wanted to cancel an already-submitted booking) must not claim
    something was discarded - that was the misleading message from the bug report."""
    result = _call(rt.cancel_reservation, {}, cfg)
    assert result["status"] == "nothing to discard"
    assert "cannot be cancelled" in result["message"]


def test_cancel_reservation_after_submission_is_also_honest(cfg, capacity):
    """Exact bug scenario: a booking was already submitted (draft cleared on success),
    then the user asks to cancel it by calling cancel_reservation again."""
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    _call(rt.submit_reservation, {}, cfg)

    result = _call(rt.cancel_reservation, {}, cfg)

    assert result["status"] == "nothing to discard"


# --- Status checks --------------------------------------------------------------------------------


def test_check_booking_status_with_no_id_and_none_on_record_asks_for_one(cfg):
    result = _call(rt.check_booking_status, {}, cfg)
    assert "booking ID" in result["error"]


def test_check_booking_status_rejects_an_invalid_id_format(cfg):
    result = _call(rt.check_booking_status, {"booking_id": "not-an-id"}, cfg)
    assert "not a valid booking ID" in result["error"]


def test_check_booking_status_reports_an_unknown_id(cfg, sql_db):
    result = _call(rt.check_booking_status, {"booking_id": "PK-ZZZZZZ"}, cfg)
    assert "No booking found" in result["error"]


def test_check_booking_status_normalises_the_id_before_looking_it_up(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)

    other = _new_cfg()
    result = _call(rt.check_booking_status, {"booking_id": f" {booking['booking_id'].lower()} "}, other)

    assert result["status"] == "pending"
    rt._clear(other["configurable"]["thread_id"])


def test_check_booking_status_reports_pending_right_after_submission(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)

    result = _call(rt.check_booking_status, {"booking_id": booking["booking_id"]}, cfg)

    assert result == {"booking_id": booking["booking_id"], "status": "pending", "decision_reason": None}


def test_check_booking_status_reflects_an_approval_with_its_reason(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    admin_db.record_decision(booking["booking_id"], "approved", "looks fine")

    result = _call(rt.check_booking_status, {"booking_id": booking["booking_id"]}, cfg)

    assert (result["status"], result["decision_reason"]) == ("approved", "looks fine")


def test_check_booking_status_without_an_id_uses_the_one_just_submitted(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)

    result = _call(rt.check_booking_status, {}, cfg)

    assert result["booking_id"] == booking["booking_id"]


def test_check_booking_status_without_an_id_uses_the_one_last_checked(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    _call(rt.check_booking_status, {}, cfg)  # remembers it from submission

    # a second conversation that explicitly checked this ID once should also be able
    # to ask again without repeating it:
    other = _new_cfg()
    _call(rt.check_booking_status, {"booking_id": booking["booking_id"]}, other)
    result = _call(rt.check_booking_status, {}, other)
    assert result["booking_id"] == booking["booking_id"]
    rt._clear(other["configurable"]["thread_id"])


def test_check_booking_status_does_not_remember_a_failed_lookup(cfg, sql_db):
    _call(rt.check_booking_status, {"booking_id": "PK-ZZZZZZ"}, cfg)
    result = _call(rt.check_booking_status, {}, cfg)
    assert "booking ID" in result["error"]  # still nothing on record, not the unknown ID


def test_check_booking_status_never_returns_more_than_id_status_and_reason(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)

    result = _call(rt.check_booking_status, {"booking_id": booking["booking_id"]}, cfg)

    assert set(result) == {"booking_id", "status", "decision_reason"}


# --- Notifying the user once a decision is made --------------------------------------------------


def test_pending_decisions_reports_nothing_while_still_pending(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    _call(rt.submit_reservation, {}, cfg)

    assert rt.pending_decisions(cfg["configurable"]["thread_id"]) == []


def test_pending_decisions_reports_an_approval_with_congrats_wording(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    admin_db.record_decision(booking["booking_id"], "approved", None)

    [decided] = rt.pending_decisions(cfg["configurable"]["thread_id"])

    assert (decided["booking_id"], decided["status"]) == (booking["booking_id"], "approved")


def test_pending_decisions_reports_a_refusal_with_its_reason(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    admin_db.record_decision(booking["booking_id"], "refused", "no space that day")

    [decided] = rt.pending_decisions(cfg["configurable"]["thread_id"])

    assert (decided["status"], decided["decision_reason"]) == ("refused", "no space that day")


def test_pending_decisions_reports_each_decided_booking_only_once(cfg, capacity):
    _fill(cfg)
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    admin_db.record_decision(booking["booking_id"], "approved", None)
    tid = cfg["configurable"]["thread_id"]

    assert len(rt.pending_decisions(tid)) == 1
    assert rt.pending_decisions(tid) == []


def test_pending_decisions_keeps_tracking_bookings_still_pending_after_a_poll(cfg, capacity):
    _fill(cfg, plates=("AAA111",))
    _call(rt.show_reservation_summary, {}, cfg)
    first = _call(rt.submit_reservation, {}, cfg)
    _fill(cfg, plates=("BBB222",))
    _call(rt.show_reservation_summary, {}, cfg)
    second = _call(rt.submit_reservation, {}, cfg)
    admin_db.record_decision(first["booking_id"], "approved", None)
    tid = cfg["configurable"]["thread_id"]

    assert [d["booking_id"] for d in rt.pending_decisions(tid)] == [first["booking_id"]]

    admin_db.record_decision(second["booking_id"], "refused", None)
    assert [d["booking_id"] for d in rt.pending_decisions(tid)] == [second["booking_id"]]


# --- Privacy -------------------------------------------------------------------------------------


def test_no_tool_other_than_the_status_check_can_look_up_a_booking_by_id():
    for agent_tool in ALL_TOOLS:
        if agent_tool.name == "check_booking_status":
            continue  # the one sanctioned exception - see its own tests below
        params = set(inspect.signature(agent_tool.func).parameters)
        assert not params & {"booking_id", "pass_id", "reservation_id"}, agent_tool.name


def test_another_conversation_cannot_see_a_booking_even_with_its_ids(cfg, capacity):
    _fill(cfg, plates=("AAA111", "BBB222"))
    _call(rt.show_reservation_summary, {}, cfg)
    booking = _call(rt.submit_reservation, {}, cfg)
    leaked_values = ["Anna", "Smith", "AAA111", "BBB222"]
    ids = [booking["booking_id"], *(p["pass_id"] for p in booking["passes"])]

    stranger = _new_cfg()
    outputs = [
        rt.update_reservation_details.invoke({"first_name": "Eve"}, stranger),
        rt.add_vehicles.invoke(_vehicles(*(i.replace("-", "") for i in ids)), stranger),
        rt.remove_vehicle.invoke({"car_number": ids[0]}, stranger),
        rt.show_reservation_summary.invoke({}, stranger),
        rt.submit_reservation.invoke({}, stranger),
        rt.cancel_reservation.invoke({}, stranger),
        # check_booking_status CAN look this ID up (by design - see its own tests),
        # but must still never reveal the booker's name or plates, only the status.
        rt.check_booking_status.invoke({"booking_id": booking["booking_id"]}, stranger),
    ]
    rt._clear(stranger["configurable"]["thread_id"])

    for output in outputs:
        for value in leaked_values:
            assert value.upper() not in output.upper()
