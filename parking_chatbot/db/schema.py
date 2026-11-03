"""Database tables, and the one-off upgrade of databases created by older versions."""
import sqlite3
from contextlib import closing
from datetime import datetime

from parking_chatbot.db.database import get_connection, immediate_transaction
from parking_chatbot.reservation.ids import new_booking_id, pass_id

RESERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reservations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id  TEXT NOT NULL UNIQUE,
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'approved', 'refused', 'expired')),
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Set once the booking has actually been handed to the admin channel (Stage 2);
    -- NULL means it exists but was never escalated, so it must stay invisible to the admin.
    escalated_at    TEXT,
    decided_at      TEXT,
    decision_reason TEXT
)"""

RESERVATION_VEHICLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reservation_vehicles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id INTEGER NOT NULL REFERENCES reservations (id) ON DELETE CASCADE,
    pass_id        TEXT NOT NULL UNIQUE,
    car_number     TEXT NOT NULL,
    -- NULL only for vehicles moved over from bookings made before vehicle types existed.
    vehicle_type   TEXT,
    UNIQUE (reservation_id, car_number)
)"""

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS working_hours (
    day_of_week TEXT PRIMARY KEY,
    open_time   TEXT NOT NULL,
    close_time  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    vehicle_type TEXT NOT NULL,
    plan         TEXT NOT NULL,
    price        REAL NOT NULL CHECK (price >= 0),
    currency     TEXT NOT NULL DEFAULT 'USD',
    description  TEXT,
    PRIMARY KEY (vehicle_type, plan)
);

CREATE TABLE IF NOT EXISTS availability (
    level            INTEGER PRIMARY KEY,
    total_spaces     INTEGER NOT NULL CHECK (total_spaces >= 0),
    available_spaces INTEGER NOT NULL CHECK (available_spaces >= 0),
    CHECK (available_spaces <= total_spaces)
);

{RESERVATIONS_TABLE_SQL};

{RESERVATION_VEHICLES_TABLE_SQL};
"""

_LEGACY_STATUS = {"pending": "pending", "confirmed": "approved", "rejected": "refused"}


def create_tables() -> None:
    """Create all tables if they do not exist yet, upgrading an older database first.

    Safe to run many times; existing reservations are moved, never deleted.
    """
    with closing(get_connection()) as conn:
        if _has_legacy_reservations(conn):
            _migrate_legacy_reservations(conn)
        conn.executescript(SCHEMA_SQL)
        added = _add_missing_columns(
            conn,
            "reservations",
            {"escalated_at": "TEXT", "decided_at": "TEXT", "decision_reason": "TEXT", "recorded_at": "TEXT"},
        )
        if "recorded_at" in added:
            _backfill_recorded_at_for_pre_stage3_approvals(conn)


def _backfill_recorded_at_for_pre_stage3_approvals(conn: sqlite3.Connection) -> None:
    """The first time recorded_at is added, every already-approved booking predates
    Stage 3 recording entirely - mark them as already recorded so the retry sweep
    doesn't send old (possibly test) approvals to the recording server the moment
    it starts running, instead of treating them as new work.

    Only ever runs once per database, right after the column is created (see the
    `"recorded_at" in added` check above) - never on an ordinary create_tables()
    call against a database that already has the column, which would otherwise
    silently mark every future approval "recorded" without it ever being sent.
    """
    with conn:
        conn.execute(
            "UPDATE reservations SET recorded_at = ? WHERE status = 'approved' AND recorded_at IS NULL",
            (datetime.now().isoformat(timespec="minutes"),),
        )


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> set[str]:
    """Add any of these columns the table doesn't already have. Safe to run every time;
    CREATE TABLE IF NOT EXISTS alone never adds columns to a table that already exists.
    Returns the names of the columns actually added this call (empty if the table
    already had all of them) - callers use this to run one-time-only follow-up work."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    missing = {name: sql_type for name, sql_type in columns.items() if name not in existing}
    if not missing:
        return set()
    with conn:
        for name, sql_type in missing.items():
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    return set(missing)


def _has_legacy_reservations(conn: sqlite3.Connection) -> bool:
    """Older versions stored one vehicle per reservation row, in a car_number column."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reservations)")}
    return "car_number" in columns


def _migrate_legacy_reservations(conn: sqlite3.Connection) -> None:
    """Move one-vehicle-per-row reservations into bookings + vehicles, in one transaction."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reservations)")}
    vehicle_type = "vehicle_type" if "vehicle_type" in columns else "NULL"
    with immediate_transaction(conn):
        conn.execute("ALTER TABLE reservations RENAME TO reservations_legacy")
        conn.execute(RESERVATIONS_TABLE_SQL)
        conn.execute(RESERVATION_VEHICLES_TABLE_SQL)
        legacy_rows = conn.execute(
            "SELECT id, first_name, last_name, car_number, "
            f"{vehicle_type}, start_time, end_time, status, created_at "
            "FROM reservations_legacy ORDER BY id"
        ).fetchall()
        used_ids: set[str] = set()
        for row_id, first, last, car_number, v_type, start, end, status, created in legacy_rows:
            booking_id = new_booking_id()
            while booking_id in used_ids:
                booking_id = new_booking_id()
            used_ids.add(booking_id)
            conn.execute(
                "INSERT INTO reservations (id, booking_id, first_name, last_name, start_time, end_time, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row_id, booking_id, first, last, start, end, _LEGACY_STATUS[status], created),
            )
            conn.execute(
                "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
                "VALUES (?, ?, ?, ?)",
                (row_id, pass_id(booking_id, 1), car_number, v_type),
            )
        conn.execute("DROP TABLE reservations_legacy")


if __name__ == "__main__":
    create_tables()
    with closing(get_connection()) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    print("Tables in database:", [r[0] for r in rows])
