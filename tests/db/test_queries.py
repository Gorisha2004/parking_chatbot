from contextlib import closing

from parking_chatbot.db import queries
from parking_chatbot.db.database import get_connection


def _seed(conn):
    conn.executemany(
        "INSERT INTO working_hours (day_of_week, open_time, close_time) VALUES (?, ?, ?)",
        [("Monday", "06:00", "23:00"), ("Sunday", "08:00", "22:00")],
    )
    conn.executemany(
        "INSERT INTO prices (vehicle_type, plan, price, currency, description) VALUES (?, ?, ?, ?, ?)",
        [
            ("car", "hourly", 3.0, "USD", "per hour"),
            ("car", "daily", 20.0, "USD", "per day"),
            ("motorcycle", "hourly", 1.0, "USD", "per hour"),
            ("van", "hourly", 5.0, "USD", "per hour"),
        ],
    )
    conn.executemany(
        "INSERT INTO availability (level, total_spaces, available_spaces) VALUES (?, ?, ?)",
        [(1, 80, 12), (2, 100, 0)],
    )


def test_get_working_hours_sorted_by_day_of_week(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    hours = queries.get_working_hours()
    assert [row["day_of_week"] for row in hours] == ["Monday", "Sunday"]


def test_get_prices_returns_every_vehicle_type_cheapest_first(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    prices = queries.get_prices()
    assert [row["price"] for row in prices] == [1.0, 3.0, 5.0, 20.0]
    assert {row["vehicle_type"] for row in prices} == {"car", "motorcycle", "van"}


def test_get_prices_filters_by_vehicle_type(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    prices = queries.get_prices("car")
    assert [(row["vehicle_type"], row["plan"]) for row in prices] == [("car", "hourly"), ("car", "daily")]


def test_get_prices_for_unknown_vehicle_type_is_empty(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    assert queries.get_prices("spaceship") == []


def test_get_vehicle_types_smallest_first(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    assert queries.get_vehicle_types() == ["motorcycle", "car", "van"]


def test_get_level_availability_known_and_unknown_level(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    assert queries.get_level_availability(2) == {
        "level": 2,
        "total_spaces": 100,
        "available_spaces": 0,
    }
    assert queries.get_level_availability(99) is None


def test_get_total_availability_sums_all_levels(sql_db):
    with closing(get_connection()) as conn, conn:
        _seed(conn)
    assert queries.get_total_availability() == {"total_spaces": 180, "available_spaces": 12}
