import re
import sqlite3
from contextlib import closing

import pytest

from parking_chatbot.db.database import get_connection
from parking_chatbot.db.schema import create_tables

BOOKING_ID = re.compile(r"^PK-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$")

ORIGINAL_LAYOUT = (
    "CREATE TABLE reservations (id INTEGER PRIMARY KEY AUTOINCREMENT, first_name TEXT NOT NULL, "
    "last_name TEXT NOT NULL, car_number TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, "
    "status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
)


def _tables() -> set[str]:
    with closing(get_connection()) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _replace_reservations_with(sql: str, rows: list[tuple]) -> None:
    """Turn the fresh test database into one created by an older version."""
    with closing(get_connection()) as conn, conn:
        conn.execute("DROP TABLE reservation_vehicles")
        conn.execute("DROP TABLE reservations")
        conn.execute(sql)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(reservations)")][1:-2]
        placeholders = ", ".join("?" * (len(columns) + 1))
        for row in rows:
            conn.execute(f"INSERT INTO reservations ({', '.join(columns)}, status) VALUES ({placeholders})", row)


def _bookings() -> list[tuple]:
    with closing(get_connection()) as conn:
        return conn.execute(
            "SELECT r.id, r.booking_id, r.first_name, r.status, v.pass_id, v.car_number, v.vehicle_type "
            "FROM reservations r JOIN reservation_vehicles v ON v.reservation_id = r.id ORDER BY r.id"
        ).fetchall()


def test_create_tables_creates_all_expected_tables(sql_db):
    assert {"working_hours", "prices", "availability", "reservations", "reservation_vehicles"} <= _tables()


def test_create_tables_is_safe_to_run_twice(sql_db):
    create_tables()
    create_tables()
    assert "reservations_legacy" not in _tables()


def test_upgrades_the_original_one_vehicle_per_row_layout(sql_db):
    _replace_reservations_with(
        ORIGINAL_LAYOUT,
        [
            ("Anna", "Smith", "AB123CD", "2026-11-20T09:00", "2026-11-20T12:00", "pending"),
            ("Ben", "Lee", "XY987ZT", "2026-11-21T09:00", "2026-11-21T12:00", "confirmed"),
        ],
    )

    create_tables()

    rows = _bookings()
    assert [(r[0], r[2], r[5], r[6]) for r in rows] == [(1, "Anna", "AB123CD", None), (2, "Ben", "XY987ZT", None)]
    assert all(BOOKING_ID.match(r[1]) for r in rows)
    assert [r[4] for r in rows] == [f"{r[1]}-01" for r in rows]
    assert "reservations_legacy" not in _tables()


def test_upgrade_keeps_vehicle_types_and_renames_statuses(sql_db):
    _replace_reservations_with(
        ORIGINAL_LAYOUT.replace("car_number TEXT NOT NULL,", "car_number TEXT NOT NULL, vehicle_type TEXT,"),
        [
            ("Anna", "Smith", "AB123CD", "van", "2026-11-20T09:00", "2026-11-20T12:00", "confirmed"),
            ("Ben", "Lee", "XY987ZT", "car", "2026-11-21T09:00", "2026-11-21T12:00", "rejected"),
        ],
    )

    create_tables()

    assert [(r[3], r[6]) for r in _bookings()] == [("approved", "van"), ("refused", "car")]


def test_a_failed_upgrade_leaves_the_old_data_untouched(sql_db):
    _replace_reservations_with(
        ORIGINAL_LAYOUT,
        [("Anna", "Smith", "AB123CD", "2026-11-20T09:00", "2026-11-20T12:00", "unknown-status")],
    )

    with pytest.raises(KeyError):
        create_tables()

    with closing(get_connection()) as conn:
        assert conn.execute("SELECT first_name, car_number FROM reservations").fetchall() == [("Anna", "AB123CD")]
    assert "reservation_vehicles" not in _tables()
    assert "reservations_legacy" not in _tables()


def test_booking_ids_and_pass_ids_must_be_unique(sql_db):
    with closing(get_connection()) as conn, conn:
        conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time) "
            "VALUES ('PK-AAAAAA', 'A', 'B', 's', 'e')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time) "
                "VALUES ('PK-AAAAAA', 'C', 'D', 's', 'e')"
            )
        conn.execute("INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number) VALUES (1, 'P-01', 'X1')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number) VALUES (1, 'P-01', 'X2')"
            )


def test_vehicles_must_belong_to_an_existing_booking(sql_db):
    with closing(get_connection()) as conn, conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number) VALUES (999, 'P-01', 'X1')"
            )


def test_availability_rejects_available_spaces_above_total(sql_db):
    with closing(get_connection()) as conn, conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, 10, 20)")
