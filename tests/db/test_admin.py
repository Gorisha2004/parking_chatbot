import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from parking_chatbot import config
from parking_chatbot.db import admin
from parking_chatbot.db.database import get_connection


def _insert_booking(
    booking_id: str,
    status: str = "pending",
    escalated: bool = True,
    created_at: datetime | None = None,
    vehicles: tuple[tuple[str, str, str], ...] = (("AB123CD", "car"),),
) -> int:
    """Insert a booking with full control over fields create_reservation() doesn't expose."""
    created = (created_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
    escalated_at = "2026-01-01T00:00" if escalated else None
    with closing(get_connection()) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, "
            "status, created_at, escalated_at) VALUES (?, 'Anna', 'Smith', "
            "'2026-11-20T09:00', '2026-11-20T12:00', ?, ?, ?)",
            (booking_id, status, created, escalated_at),
        )
        reservation_id = cursor.lastrowid
        for number, (car_number, vehicle_type) in enumerate(vehicles, start=1):
            conn.execute(
                "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
                "VALUES (?, ?, ?, ?)",
                (reservation_id, f"{booking_id}-{number:02d}", car_number, vehicle_type),
            )
    return reservation_id


# --- get_pending_request ---------------------------------------------------------------------------


def test_get_pending_request_returns_full_details_for_a_pending_escalated_booking(sql_db):
    _insert_booking("PK-AAAAAA", vehicles=(("AAA111", "car"),))

    request = admin.get_pending_request("PK-AAAAAA")

    assert request.booking_id == "PK-AAAAAA"
    assert [v.car_number for v in request.vehicles] == ["AAA111"]


@pytest.mark.parametrize(
    "kwargs", [{"status": "approved"}, {"escalated": False}], ids=["already-decided", "not-escalated"]
)
def test_get_pending_request_returns_none_for_a_booking_that_is_not_awaiting_decision(sql_db, kwargs):
    _insert_booking("PK-AAAAAA", **kwargs)
    assert admin.get_pending_request("PK-AAAAAA") is None


def test_get_pending_request_returns_none_for_an_unknown_id(sql_db):
    assert admin.get_pending_request("PK-ZZZZZZ") is None


# --- list_pending_bookings ------------------------------------------------------------------------


def test_list_pending_bookings_excludes_non_escalated_and_non_pending(sql_db):
    _insert_booking("PK-AAAAAA", status="pending", escalated=True)
    _insert_booking("PK-BBBBBB", status="pending", escalated=False)
    _insert_booking("PK-CCCCCC", status="approved", escalated=True)

    pending = admin.list_pending_bookings()

    assert [r.booking_id for r in pending] == ["PK-AAAAAA"]


def test_list_pending_bookings_includes_every_vehicle_in_order(sql_db):
    _insert_booking("PK-AAAAAA", vehicles=(("AAA111", "car"), ("BBB222", "van")))

    [request] = admin.list_pending_bookings()

    assert [(v.pass_id, v.car_number, v.vehicle_type) for v in request.vehicles] == [
        ("PK-AAAAAA-01", "AAA111", "car"),
        ("PK-AAAAAA-02", "BBB222", "van"),
    ]


# --- get_booking -----------------------------------------------------------------------------------


def test_get_booking_returns_none_for_an_unknown_id(sql_db):
    assert admin.get_booking("PK-NOPE00") is None


def test_get_booking_returns_status_and_vehicles(sql_db):
    _insert_booking("PK-AAAAAA", status="approved")

    result = admin.get_booking("PK-AAAAAA")

    assert result["status"] == "approved"
    assert result["vehicles"] == [{"pass_id": "PK-AAAAAA-01", "car_number": "AB123CD", "vehicle_type": "car"}]


def test_get_booking_flips_an_overdue_pending_booking_to_expired(sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    overdue = datetime.now(timezone.utc) - timedelta(minutes=61)
    _insert_booking("PK-AAAAAA", status="pending", created_at=overdue)

    result = admin.get_booking("PK-AAAAAA")

    assert result["status"] == "expired"
    # and it is persisted, not just reported once:
    with closing(get_connection()) as conn:
        assert conn.execute("SELECT status FROM reservations WHERE booking_id = 'PK-AAAAAA'").fetchone() == (
            "expired",
        )


def test_get_booking_does_not_expire_a_booking_still_within_the_window(sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    _insert_booking("PK-AAAAAA", status="pending", created_at=recent)

    assert admin.get_booking("PK-AAAAAA")["status"] == "pending"


# --- record_decision --------------------------------------------------------------------------------


def test_record_decision_approves_and_stores_the_time_and_reason(sql_db):
    _insert_booking("PK-AAAAAA")

    admin.record_decision("PK-AAAAAA", "approved", "looks fine")

    result = admin.get_booking("PK-AAAAAA")
    assert result["status"] == "approved"
    assert result["decision_reason"] == "looks fine"
    datetime.fromisoformat(result["decided_at"])  # stored as a parseable local timestamp


def test_record_decision_refuses_without_a_reason(sql_db):
    _insert_booking("PK-AAAAAA")
    admin.record_decision("PK-AAAAAA", "refused", None)
    assert admin.get_booking("PK-AAAAAA")["status"] == "refused"


def test_record_decision_on_an_unknown_booking_raises(sql_db):
    with pytest.raises(admin.UnknownBookingError):
        admin.record_decision("PK-NOPE00", "approved", None)


def test_record_decision_twice_raises_and_keeps_the_first_decision(sql_db):
    _insert_booking("PK-AAAAAA")
    admin.record_decision("PK-AAAAAA", "approved", "first")

    with pytest.raises(admin.AlreadyDecidedError):
        admin.record_decision("PK-AAAAAA", "refused", "second")

    assert admin.get_booking("PK-AAAAAA")["decision_reason"] == "first"


def test_record_decision_on_an_expired_booking_raises_already_decided(sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    overdue = datetime.now(timezone.utc) - timedelta(minutes=61)
    _insert_booking("PK-AAAAAA", status="pending", created_at=overdue)

    with pytest.raises(admin.AlreadyDecidedError) as error:
        admin.record_decision("PK-AAAAAA", "approved", None)
    assert error.value.status == "expired"


def test_two_simultaneous_decisions_on_the_same_booking_cannot_both_win(sql_db):
    _insert_booking("PK-AAAAAA")
    outcomes = []

    def decide(decision):
        try:
            admin.record_decision("PK-AAAAAA", decision, None)
            outcomes.append(("ok", decision))
        except admin.AlreadyDecidedError:
            outcomes.append(("rejected", decision))

    threads = [threading.Thread(target=decide, args=(d,)) for d in ("approved", "refused")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(o[0] for o in outcomes) == ["ok", "rejected"]
    final_status = admin.get_booking("PK-AAAAAA")["status"]
    assert final_status == next(d for outcome, d in outcomes if outcome == "ok")
