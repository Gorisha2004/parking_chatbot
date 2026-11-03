from datetime import datetime

import pytest

from parking_chatbot import config
from parking_chatbot.db import admin as admin_db
from parking_chatbot.recording import client


def _insert_approved(booking_id="PK-AAAAAA"):
    from contextlib import closing

    from parking_chatbot.db.database import get_connection

    with closing(get_connection()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, "
            "status, escalated_at, decided_at) VALUES (?, 'Anna', 'Smith', '2026-11-20T09:00', "
            "'2026-11-20T12:00', 'approved', '2026-01-01T00:00', '2026-11-19T14:32')",
            (booking_id,),
        )
        conn.execute(
            "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
            "VALUES (?, ?, 'AB123CD', 'car')",
            (cur.lastrowid, f"{booking_id}-01"),
        )


def test_mcp_endpoint_url_appends_mcp_to_a_bare_origin(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVER_URL", "http://127.0.0.1:8002")
    assert client._mcp_endpoint_url() == "http://127.0.0.1:8002/mcp"


def test_mcp_endpoint_url_does_not_double_append_when_env_already_has_it(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVER_URL", "http://127.0.0.1:8002/mcp")
    assert client._mcp_endpoint_url() == "http://127.0.0.1:8002/mcp"


def test_mcp_endpoint_url_handles_a_trailing_slash(monkeypatch):
    monkeypatch.setattr(config, "MCP_SERVER_URL", "http://127.0.0.1:8002/")
    assert client._mcp_endpoint_url() == "http://127.0.0.1:8002/mcp"


def test_to_utc_iso_converts_facility_local_time():
    result = client._to_utc_iso(datetime(2026, 11, 19, 14, 32))
    assert result == "2026-11-19T14:32:00+00:00"  # FACILITY_TIMEZONE default is UTC


async def _ok(booking, timeout):
    return {"recorded": True}


async def _boom(booking, timeout):
    raise RuntimeError("server down")


def test_send_approved_booking_marks_recorded_on_success(sql_db, monkeypatch):
    _insert_approved("PK-AAAAAA")
    monkeypatch.setattr(client, "_call_record_reservation", _ok)

    client.send_approved_booking("PK-AAAAAA")

    assert admin_db.list_unrecorded_approved_bookings() == []


def test_send_approved_booking_never_raises_on_failure(sql_db, monkeypatch):
    _insert_approved("PK-AAAAAA")
    monkeypatch.setattr(client, "_call_record_reservation", _boom)

    client.send_approved_booking("PK-AAAAAA")  # must not raise

    [pending] = admin_db.list_unrecorded_approved_bookings()
    assert pending.booking_id == "PK-AAAAAA"


def test_send_approved_booking_does_nothing_for_an_unknown_booking(sql_db, monkeypatch):
    async def must_not_be_called(booking, timeout):
        pytest.fail("must not be called")

    monkeypatch.setattr(client, "_call_record_reservation", must_not_be_called)
    client.send_approved_booking("PK-NOPE00")  # must not raise


def test_retry_unrecorded_marks_each_successful_booking(sql_db, monkeypatch):
    _insert_approved("PK-AAAAAA")
    _insert_approved("PK-BBBBBB")
    monkeypatch.setattr(client, "_call_record_reservation", _ok)

    client.retry_unrecorded()

    assert admin_db.list_unrecorded_approved_bookings() == []


def test_retry_unrecorded_continues_past_a_failure(sql_db, monkeypatch):
    _insert_approved("PK-AAAAAA")
    _insert_approved("PK-BBBBBB")

    async def flaky(booking, timeout):
        if booking.booking_id == "PK-AAAAAA":
            raise RuntimeError("server down")
        return {"recorded": True}

    monkeypatch.setattr(client, "_call_record_reservation", flaky)

    client.retry_unrecorded()  # must not raise

    remaining = {b.booking_id for b in admin_db.list_unrecorded_approved_bookings()}
    assert remaining == {"PK-AAAAAA"}


def test_apply_decision_calls_the_hook_only_on_approval(sql_db, monkeypatch):
    from contextlib import closing

    from parking_chatbot.admin import agent
    from parking_chatbot.db.database import get_connection

    calls = []
    monkeypatch.setattr("parking_chatbot.recording.client.send_approved_booking", lambda booking_id: calls.append(booking_id))

    with closing(get_connection()) as conn, conn:
        conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, escalated_at) "
            "VALUES ('PK-X', 'Anna', 'Smith', '2026-11-20T09:00', '2026-11-20T12:00', '2026-01-01T00:00')"
        )
        conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, escalated_at) "
            "VALUES ('PK-Y', 'Anna', 'Smith', '2026-11-20T09:00', '2026-11-20T12:00', '2026-01-01T00:00')"
        )

    agent.apply_decision("PK-X", "refused", None)
    assert calls == []

    agent.apply_decision("PK-Y", "approved", None)
    assert calls == ["PK-Y"]


def test_start_retry_sweep_thread_runs_immediately_and_stops_cleanly(sql_db, monkeypatch):
    monkeypatch.setattr(config, "MCP_RETRY_INTERVAL_SECONDS", 5)
    calls = []
    monkeypatch.setattr(client, "retry_unrecorded", lambda: calls.append(1))

    stop_event = client.start_retry_sweep_thread()
    import time

    time.sleep(0.2)
    stop_event.set()

    assert calls  # ran at least once, immediately, without waiting for the interval


def test_start_retry_sweep_thread_survives_a_failing_sweep(sql_db, monkeypatch):
    monkeypatch.setattr(config, "MCP_RETRY_INTERVAL_SECONDS", 5)

    def boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(client, "retry_unrecorded", boom)

    stop_event = client.start_retry_sweep_thread()  # must not crash the process
    import time

    time.sleep(0.2)
    stop_event.set()
