"""The vehicle types are defined once (reservation.models.VEHICLE_TYPES); everything
that repeats them for the LLM or the database must agree with that list."""
import importlib.util

from parking_chatbot import config
from parking_chatbot.chatbot.reservation_tools import add_vehicles
from parking_chatbot.chatbot.tools import lookup_prices
from parking_chatbot.reservation.models import VEHICLE_TYPES


def _seed_module():
    spec = importlib.util.spec_from_file_location("seed_db", config.PROJECT_ROOT / "scripts" / "seed_db.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_vehicle_type_has_seeded_prices():
    assert tuple(_seed_module().PRICES_BY_VEHICLE) == VEHICLE_TYPES


def test_price_tool_description_names_every_vehicle_type():
    for vehicle_type in VEHICLE_TYPES:
        assert vehicle_type in lookup_prices.description


def test_add_vehicles_tool_description_names_every_vehicle_type():
    for vehicle_type in VEHICLE_TYPES:
        assert vehicle_type in add_vehicles.description
