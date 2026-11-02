import importlib.util
from contextlib import closing

from parking_chatbot import config
from parking_chatbot.db import queries
from parking_chatbot.db.database import get_connection

PLANS = {"hourly", "daily", "weekly", "monthly"}
SIZE_ORDER = ["motorcycle", "three_wheeler", "car", "van"]


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("seed_db", config.PROJECT_ROOT / "scripts" / "seed_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed_db = _load_seed_module()


def test_seed_loads_hours_prices_and_availability(sql_db):
    seed_db.seed()
    assert len(queries.get_working_hours()) == 7
    assert len(queries.get_availability()) == 4
    for vehicle_type in SIZE_ORDER:
        assert {row["plan"] for row in queries.get_prices(vehicle_type)} == PLANS


def test_bigger_vehicles_pay_more_on_every_plan(sql_db):
    seed_db.seed()
    price = {(r["vehicle_type"], r["plan"]): r["price"] for r in queries.get_prices()}
    for plan in PLANS:
        by_size = [price[(vehicle_type, plan)] for vehicle_type in SIZE_ORDER]
        assert by_size == sorted(by_size) and len(set(by_size)) == len(by_size), plan


def test_seed_is_idempotent(sql_db):
    seed_db.seed()
    seed_db.seed()
    assert len(queries.get_working_hours()) == 7
    assert len(queries.get_prices()) == len(SIZE_ORDER) * len(PLANS)
    assert len(queries.get_availability()) == 4


def test_seed_upgrades_an_old_price_table_and_keeps_reservations(sql_db):
    with closing(get_connection()) as conn, conn:
        conn.execute("DROP TABLE prices")
        conn.execute("CREATE TABLE prices (plan TEXT PRIMARY KEY, price REAL, currency TEXT, description TEXT)")
        conn.execute("INSERT INTO prices VALUES ('hourly', 3.0, 'USD', 'old')")
        conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time) "
            "VALUES ('PK-AAAAAA', 'Anna', 'Smith', '2026-11-01T09:00', '2026-11-01T12:00')"
        )

    seed_db.seed()

    assert queries.get_vehicle_types() == SIZE_ORDER
    with closing(get_connection()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
