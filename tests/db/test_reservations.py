import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta

import pytest

from parking_chatbot.db import reservations
from parking_chatbot.db.database import get_connection
from parking_chatbot.db.reservations import NotEnoughSpacesError, create_reservation, get_free_spaces
from parking_chatbot.reservation.models import ReservationDraft, Vehicle

BOOKING_ID = re.compile(r"^PK-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$")
START = (datetime.now() + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
END = START + timedelta(hours=8)


def _set_capacity(total: int) -> None:
    with closing(get_connection()) as conn, conn:
        conn.execute("DELETE FROM availability")
        conn.execute("INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, ?, ?)", (total, total))


def _draft(vehicles: int, start: datetime = START, end: datetime = END, prefix: str = "CAR") -> ReservationDraft:
    return ReservationDraft(
        first_name="Anna",
        last_name="Smith",
        vehicles=[Vehicle(car_number=f"{prefix}{i:03d}", vehicle_type="car") for i in range(1, vehicles + 1)],
        start=start,
        end=end,
    )


def _count(table: str) -> int:
    with closing(get_connection()) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture(autouse=True)
def capacity(sql_db):
    _set_capacity(100)


def test_saves_a_pending_group_booking_with_one_pass_per_vehicle():
    saved = create_reservation(_draft(3))

    assert BOOKING_ID.match(saved.booking_id)
    assert [p.pass_id for p in saved.passes] == [f"{saved.booking_id}-0{n}" for n in (1, 2, 3)]
    with closing(get_connection()) as conn:
        booking = conn.execute("SELECT booking_id, first_name, status FROM reservations").fetchone()
        stored = conn.execute("SELECT pass_id, car_number, vehicle_type FROM reservation_vehicles ORDER BY id").fetchall()
    assert booking == (saved.booking_id, "Anna", "pending")
    assert stored == [(p.pass_id, p.car_number, p.vehicle_type) for p in saved.passes]


def test_rejects_an_incomplete_draft():
    with pytest.raises(ValueError):
        create_reservation(ReservationDraft(first_name="Anna"))


def test_each_booking_gets_a_different_id():
    first = create_reservation(_draft(1, prefix="AAA"))
    second = create_reservation(_draft(1, prefix="BBB"))
    assert first.booking_id != second.booking_id


def test_retries_with_a_new_id_when_the_generated_one_already_exists(monkeypatch):
    existing = create_reservation(_draft(1, prefix="AAA")).booking_id
    generated = iter([existing, "PK-NEWNEW"])
    monkeypatch.setattr(reservations, "new_booking_id", lambda: next(generated))

    assert create_reservation(_draft(1, prefix="BBB")).booking_id == "PK-NEWNEW"


def test_gives_up_after_repeated_id_clashes(monkeypatch):
    existing = create_reservation(_draft(1, prefix="AAA")).booking_id
    monkeypatch.setattr(reservations, "new_booking_id", lambda: existing)

    with pytest.raises(RuntimeError, match="unique booking ID"):
        create_reservation(_draft(1, prefix="BBB"))
    assert _count("reservations") == 1


def test_counts_every_vehicle_of_the_group_against_capacity():
    _set_capacity(4)
    with pytest.raises(NotEnoughSpacesError) as error:
        create_reservation(_draft(5))
    assert (error.value.free, error.value.needed) == (4, 5)
    assert _count("reservations") == 0


def test_overlapping_bookings_use_up_capacity_for_that_period():
    _set_capacity(5)
    create_reservation(_draft(3, prefix="AAA"))

    assert get_free_spaces(START, END) == 2
    with pytest.raises(NotEnoughSpacesError):
        create_reservation(_draft(3, prefix="BBB"))


def test_bookings_that_do_not_overlap_do_not_count():
    _set_capacity(3)
    create_reservation(_draft(3, prefix="AAA"))

    assert get_free_spaces(END, END + timedelta(hours=2)) == 3
    assert get_free_spaces(START - timedelta(hours=2), START) == 3


@pytest.mark.parametrize("status", ["refused", "expired"])
def test_refused_and_expired_bookings_free_their_spaces(status):
    _set_capacity(3)
    create_reservation(_draft(3, prefix="AAA"))
    with closing(get_connection()) as conn, conn:
        conn.execute("UPDATE reservations SET status = ?", (status,))

    assert get_free_spaces(START, END) == 3


def test_two_simultaneous_bookings_cannot_both_take_the_last_spaces():
    _set_capacity(3)
    barrier = threading.Barrier(2)
    outcomes = []

    def book(prefix):
        barrier.wait()
        try:
            create_reservation(_draft(2, prefix=prefix))
            outcomes.append("saved")
        except NotEnoughSpacesError:
            outcomes.append("refused")

    threads = [threading.Thread(target=book, args=(prefix,)) for prefix in ("AAA", "BBB")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["refused", "saved"]
    assert _count("reservation_vehicles") == 2


def test_a_failure_part_way_through_saves_nothing(monkeypatch):
    monkeypatch.setattr(reservations, "pass_id", lambda booking_id, number: "SAME-PASS")

    with pytest.raises(sqlite3.IntegrityError):
        create_reservation(_draft(2))
    assert _count("reservations") == 0
    assert _count("reservation_vehicles") == 0
