from contextlib import closing

from parking_chatbot.db.database import get_connection
from parking_chatbot.db.schema import create_tables

WORKING_HOURS = [
    ("Monday", "06:00", "23:00"),
    ("Tuesday", "06:00", "23:00"),
    ("Wednesday", "06:00", "23:00"),
    ("Thursday", "06:00", "23:00"),
    ("Friday", "06:00", "23:00"),
    ("Saturday", "07:00", "23:00"),
    ("Sunday", "08:00", "22:00"),
]

PLAN_DESCRIPTIONS = {
    "hourly": "Price per started hour",
    "daily": "Price for 24 hours",
    "weekly": "Price for 7 days",
    "monthly": "Price for 30 days",
}

# Bigger vehicles take more space, so they pay more (hourly, daily, weekly, monthly).
PRICES_BY_VEHICLE = {
    "motorcycle": (1.00, 7.00, 40.00, 120.00),
    "three_wheeler": (2.00, 12.00, 70.00, 220.00),
    "car": (3.00, 20.00, 110.00, 350.00),
    "van": (5.00, 32.00, 180.00, 560.00),
}

PRICES = [
    (vehicle_type, plan, price, "USD", PLAN_DESCRIPTIONS[plan])
    for vehicle_type, plan_prices in PRICES_BY_VEHICLE.items()
    for plan, price in zip(PLAN_DESCRIPTIONS, plan_prices)
]

AVAILABILITY = [
    (1, 80, 12),
    (2, 100, 35),
    (3, 100, 0),
    (4, 90, 60),
]


def seed() -> None:
    """Create tables (if needed) and load demo data. Safe to run repeatedly.

    Reservations are never touched; only the reference tables are (re)loaded.
    """
    with closing(get_connection()) as conn, conn:
        # Rebuilt rather than upserted so an older table layout or removed prices can't linger.
        conn.execute("DROP TABLE IF EXISTS prices")
    create_tables()
    with closing(get_connection()) as conn, conn:
        conn.executemany(
            "INSERT OR REPLACE INTO working_hours (day_of_week, open_time, close_time) "
            "VALUES (?, ?, ?)",
            WORKING_HOURS,
        )
        conn.executemany(
            "INSERT INTO prices (vehicle_type, plan, price, currency, description) "
            "VALUES (?, ?, ?, ?, ?)",
            PRICES,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO availability (level, total_spaces, available_spaces) "
            "VALUES (?, ?, ?)",
            AVAILABILITY,
        )
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("working_hours", "prices", "availability")
        }
    print("Seeded rows per table:", counts)


if __name__ == "__main__":
    seed()