"""Saving reservations, and the capacity check that guards it."""
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime

from parking_chatbot.db.database import get_connection, immediate_transaction
from parking_chatbot.reservation.ids import new_booking_id, pass_id
from parking_chatbot.reservation.models import ReservationDraft

ACTIVE_STATUSES = ("pending", "approved")
MAX_ID_ATTEMPTS = 5


class NotEnoughSpacesError(Exception):
    """The booking has more vehicles than there are free spaces for its period."""

    def __init__(self, free: int, needed: int):
        super().__init__(f"Only {free} spaces are free for that period, but the booking has {needed} vehicles.")
        self.free = free
        self.needed = needed


@dataclass(frozen=True)
class VehiclePass:
    pass_id: str
    car_number: str
    vehicle_type: str


@dataclass(frozen=True)
class SavedReservation:
    booking_id: str
    passes: list[VehiclePass]


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="minutes")


def _free_spaces(conn: sqlite3.Connection, start: datetime, end: datetime) -> int:
    """Total spaces minus vehicles in pending or approved bookings that overlap [start, end)."""
    capacity = conn.execute("SELECT COALESCE(SUM(total_spaces), 0) FROM availability").fetchone()[0]
    statuses = ", ".join("?" * len(ACTIVE_STATUSES))
    booked = conn.execute(
        "SELECT COUNT(*) FROM reservation_vehicles v JOIN reservations r ON r.id = v.reservation_id "
        f"WHERE r.status IN ({statuses}) AND r.start_time < ? AND r.end_time > ?",
        (*ACTIVE_STATUSES, _iso(end), _iso(start)),
    ).fetchone()[0]
    return max(capacity - booked, 0)


def get_free_spaces(start: datetime, end: datetime) -> int:
    """Free spaces for a future period, counting every vehicle of overlapping active bookings."""
    with closing(get_connection()) as conn:
        return _free_spaces(conn, start, end)


def _insert_booking(conn: sqlite3.Connection, draft: ReservationDraft) -> tuple[int, str]:
    """Insert the booking row under a fresh random ID, retrying on the rare ID clash."""
    for _ in range(MAX_ID_ATTEMPTS):
        booking_id = new_booking_id()
        try:
            cursor = conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?)",
                (booking_id, draft.first_name, draft.last_name, _iso(draft.start), _iso(draft.end)),
            )
            return cursor.lastrowid, booking_id
        except sqlite3.IntegrityError as error:
            if "reservations.booking_id" not in str(error):
                raise
    raise RuntimeError(f"Could not generate a unique booking ID in {MAX_ID_ATTEMPTS} attempts.")


def create_reservation(draft: ReservationDraft) -> SavedReservation:
    """Save a complete draft as one 'pending' booking with a pass per vehicle.

    The capacity check and all inserts run in one transaction: either the whole
    booking is saved, or nothing is.
    """
    if not draft.is_complete():
        raise ValueError(f"Cannot save an incomplete reservation, missing: {draft.missing_fields()}")

    with closing(get_connection()) as conn, immediate_transaction(conn):
        free = _free_spaces(conn, draft.start, draft.end)
        if free < len(draft.vehicles):
            raise NotEnoughSpacesError(free, len(draft.vehicles))
        row_id, booking_id = _insert_booking(conn, draft)
        passes = []
        for number, vehicle in enumerate(draft.vehicles, start=1):
            vehicle_pass = VehiclePass(pass_id(booking_id, number), vehicle.car_number, vehicle.vehicle_type)
            conn.execute(
                "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
                "VALUES (?, ?, ?, ?)",
                (row_id, vehicle_pass.pass_id, vehicle_pass.car_number, vehicle_pass.vehicle_type),
            )
            passes.append(vehicle_pass)
    return SavedReservation(booking_id, passes)
