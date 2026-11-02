"""Database access for Stage 2: reading bookings awaiting a decision and recording one.

Expiry is lazy - checked whenever a booking is read, by comparing against
config.ADMIN_REQUEST_EXPIRY_MINUTES - rather than needing a background job.
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from parking_chatbot import config
from parking_chatbot.admin.models import ApprovalRequest, RequestVehicle
from parking_chatbot.db.database import get_connection, immediate_transaction


class AdminBookingError(Exception):
    """Base class for booking-decision errors, so callers can catch either kind at once."""


class UnknownBookingError(AdminBookingError):
    """No booking exists with this booking_id."""

    def __init__(self, booking_id: str):
        super().__init__(f"No booking found with ID {booking_id}.")
        self.booking_id = booking_id


class AlreadyDecidedError(AdminBookingError):
    """This booking was already approved or refused; the same request cannot be
    decided twice. Distinct from ExpiredBookingError below: this means someone
    actually decided, not that nobody did in time."""

    def __init__(self, booking_id: str, status: str, decided_at: str | None = None, decision_reason: str | None = None):
        when = f" on {decided_at}" if decided_at else ""
        reason_note = f" (reason: {decision_reason})" if decision_reason else ""
        super().__init__(f"Booking {booking_id} is already {status}{when}{reason_note}.")
        self.booking_id = booking_id
        self.status = status
        self.decided_at = decided_at
        self.decision_reason = decision_reason


class ExpiredBookingError(AdminBookingError):
    """This booking was pending but nobody decided before ADMIN_REQUEST_EXPIRY_MINUTES
    ran out. Distinct from AlreadyDecidedError: there is no decision to report."""

    def __init__(self, booking_id: str):
        super().__init__(f"Booking {booking_id} expired before a decision was made.")
        self.booking_id = booking_id


def _expire_overdue(conn: sqlite3.Connection) -> None:
    """Flip pending bookings past the expiry window to 'expired'.

    created_at is SQLite's CURRENT_TIMESTAMP, which is UTC in 'YYYY-MM-DD HH:MM:SS'
    format - the cutoff below is built the same way so the two compare correctly.
    This is the one place in the app that deals in UTC; everywhere else (start_time,
    end_time, decided_at) is local time, matching what the user actually typed.

    Deliberately does not commit: callers inside record_decision's immediate_transaction
    need this update to be part of that one transaction, not committed early on its own
    (which would end that transaction before record_decision's own write runs).
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=config.ADMIN_REQUEST_EXPIRY_MINUTES)
    conn.execute(
        "UPDATE reservations SET status = 'expired' WHERE status = 'pending' AND created_at < ?",
        (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
    )


def _vehicles_for(conn: sqlite3.Connection, reservation_id: int) -> list[RequestVehicle]:
    rows = conn.execute(
        "SELECT pass_id, car_number, vehicle_type FROM reservation_vehicles "
        "WHERE reservation_id = ? ORDER BY id",
        (reservation_id,),
    ).fetchall()
    return [RequestVehicle(pass_id=r[0], car_number=r[1], vehicle_type=r[2]) for r in rows]


def list_pending_bookings() -> list[ApprovalRequest]:
    """Bookings actually handed to the admin channel (escalated_at set) and still pending."""
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        _expire_overdue(conn)
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM reservations WHERE status = 'pending' AND escalated_at IS NOT NULL "
            "ORDER BY created_at"
        ).fetchall()
        return [
            ApprovalRequest(
                booking_id=row["booking_id"],
                first_name=row["first_name"],
                last_name=row["last_name"],
                start=datetime.fromisoformat(row["start_time"]),
                end=datetime.fromisoformat(row["end_time"]),
                vehicles=_vehicles_for(conn, row["id"]),
            )
            for row in rows
        ]


def get_pending_request(booking_id: str) -> ApprovalRequest | None:
    """The full details of one booking (booker, period, vehicles), if it is currently
    pending and escalated - None otherwise (unknown, already decided, or expired).
    Unlike get_booking(), this is for the administrator reviewing it, not the status-
    only check the chatbot exposes to customers."""
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        _expire_overdue(conn)
        conn.commit()
        row = conn.execute(
            "SELECT * FROM reservations WHERE booking_id = ? AND status = 'pending' AND escalated_at IS NOT NULL",
            (booking_id,),
        ).fetchone()
        if row is None:
            return None
        return ApprovalRequest(
            booking_id=row["booking_id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            start=datetime.fromisoformat(row["start_time"]),
            end=datetime.fromisoformat(row["end_time"]),
            vehicles=_vehicles_for(conn, row["id"]),
        )


def get_booking_for_admin(booking_id: str) -> dict | None:
    """The full details AND the real, current status of one booking, regardless of
    status (pending, approved, refused or expired) - None if the ID doesn't exist.

    For the administrator, who may see everything about any booking they ask
    about - unlike get_booking() (status-only, for the customer-facing
    check_booking_status) and get_pending_request() (full details, but pending only).
    """
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        _expire_overdue(conn)
        conn.commit()
        row = conn.execute("SELECT * FROM reservations WHERE booking_id = ?", (booking_id,)).fetchone()
        if row is None:
            return None
        return {
            "booking_id": row["booking_id"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "start": row["start_time"],
            "end": row["end_time"],
            "status": row["status"],
            "decided_at": row["decided_at"],
            "decision_reason": row["decision_reason"],
            "vehicles": [v.model_dump() for v in _vehicles_for(conn, row["id"])],
        }


def get_booking(booking_id: str) -> dict | None:
    """The current status of one booking, or None if the ID doesn't exist."""
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        _expire_overdue(conn)
        conn.commit()
        row = conn.execute("SELECT * FROM reservations WHERE booking_id = ?", (booking_id,)).fetchone()
        if row is None:
            return None
        return {
            "booking_id": row["booking_id"],
            "status": row["status"],
            "decided_at": row["decided_at"],
            "decision_reason": row["decision_reason"],
            "vehicles": [v.model_dump() for v in _vehicles_for(conn, row["id"])],
        }


def mark_escalated(booking_id: str) -> None:
    """Record that this booking was handed to the admin channel. Idempotent: a second
    call keeps the original timestamp rather than overwriting it."""
    with closing(get_connection()) as conn, conn:
        conn.execute(
            "UPDATE reservations SET escalated_at = COALESCE(escalated_at, ?) WHERE booking_id = ?",
            (datetime.now().isoformat(timespec="minutes"), booking_id),
        )


def record_decision(booking_id: str, decision: str, reason: str | None) -> None:
    """Persist the administrator's decision.

    Raises UnknownBookingError for a booking_id that doesn't exist, ExpiredBookingError
    if it was pending but ran out the clock, or AlreadyDecidedError (with the existing
    decision and when, so callers can report what actually happened) if someone
    already decided - so the same request can never be processed twice.
    """
    with closing(get_connection()) as conn, immediate_transaction(conn):
        _expire_overdue(conn)
        row = conn.execute(
            "SELECT status, decided_at, decision_reason FROM reservations WHERE booking_id = ?", (booking_id,)
        ).fetchone()
        if row is None:
            raise UnknownBookingError(booking_id)
        status, decided_at, decision_reason = row
        if status == "expired":
            raise ExpiredBookingError(booking_id)
        if status != "pending":
            raise AlreadyDecidedError(booking_id, status, decided_at, decision_reason)

        decided_at = datetime.now().isoformat(timespec="minutes")  # local time, like start_time/end_time
        conn.execute(
            "UPDATE reservations SET status = ?, decided_at = ?, decision_reason = ? WHERE booking_id = ?",
            (decision, decided_at, reason, booking_id),
        )
