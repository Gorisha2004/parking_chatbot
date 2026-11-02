import inspect
import json
import re
import uuid
from contextlib import closing
from datetime import datetime, timedelta

import pytest

from parking_chatbot import config
from parking_chatbot.chatbot import reservation_tools as rt
from parking_chatbot.chatbot.tools import ALL_TOOLS
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


# --- Privacy -------------------------------------------------------------------------------------


def test_no_tool_can_look_up_a_booking_by_id():
    for agent_tool in ALL_TOOLS:
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
    ]
    rt._clear(stranger["configurable"]["thread_id"])

    for output in outputs:
        for value in leaked_values:
            assert value.upper() not in output.upper()
