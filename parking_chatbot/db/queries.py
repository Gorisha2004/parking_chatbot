import sqlite3
from contextlib import closing

from parking_chatbot.db.database import get_connection

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _fetch_all(query: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return the rows as a list of plain dictionaries."""
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_working_hours() -> list[dict]:
    rows = _fetch_all("SELECT day_of_week, open_time, close_time FROM working_hours")
    return sorted(rows, key=lambda row: DAYS.index(row["day_of_week"]))


def get_prices(vehicle_type: str | None = None) -> list[dict]:
    """Prices per vehicle type and plan, cheapest first; optionally for one vehicle type."""
    query = "SELECT vehicle_type, plan, price, currency, description FROM prices"
    params: tuple = ()
    if vehicle_type is not None:
        query += " WHERE vehicle_type = ?"
        params = (vehicle_type,)
    return _fetch_all(query + " ORDER BY price", params)


def get_vehicle_types() -> list[str]:
    """Vehicle types that have prices, smallest (cheapest) first."""
    rows = _fetch_all("SELECT vehicle_type FROM prices GROUP BY vehicle_type ORDER BY MIN(price)")
    return [row["vehicle_type"] for row in rows]


def get_availability() -> list[dict]:
    return _fetch_all(
        "SELECT level, total_spaces, available_spaces FROM availability ORDER BY level"
    )


def get_level_availability(level: int) -> dict | None:
    rows = _fetch_all(
        "SELECT level, total_spaces, available_spaces FROM availability WHERE level = ?",
        (level,),
    )
    return rows[0] if rows else None


def get_total_availability() -> dict:
    levels = get_availability()
    return {
        "total_spaces": sum(row["total_spaces"] for row in levels),
        "available_spaces": sum(row["available_spaces"] for row in levels),
    }


if __name__ == "__main__":
    print(get_working_hours())
    print(get_prices())
    print(get_availability())
    print(get_level_availability(3))
    print(get_level_availability(99))
    print(get_total_availability())