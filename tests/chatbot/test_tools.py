import json

from parking_chatbot.chatbot import tools
from parking_chatbot.db import queries


def test_search_parking_info_uses_rag_retrieval(monkeypatch):
    monkeypatch.setattr(tools, "retrieve", lambda question: ["doc1", "doc2"])
    monkeypatch.setattr(tools, "format_context", lambda docs: f"formatted:{docs}")

    result = tools.search_parking_info.invoke({"question": "Is there EV charging?"})

    assert result == "formatted:['doc1', 'doc2']"


def test_lookup_working_hours_returns_db_result_as_json(monkeypatch):
    monkeypatch.setattr(
        queries, "get_working_hours", lambda: [{"day_of_week": "Monday", "open_time": "06:00", "close_time": "23:00"}]
    )
    result = tools.lookup_working_hours.invoke({})
    assert json.loads(result) == [{"day_of_week": "Monday", "open_time": "06:00", "close_time": "23:00"}]


def test_lookup_prices_without_vehicle_type_returns_all_prices(monkeypatch):
    calls = []
    all_prices = [{"vehicle_type": "car", "plan": "hourly", "price": 3.0}]
    monkeypatch.setattr(queries, "get_prices", lambda vehicle_type: calls.append(vehicle_type) or all_prices)

    result = tools.lookup_prices.invoke({})

    assert json.loads(result) == all_prices
    assert calls == [None]


def test_lookup_prices_passes_vehicle_type_to_the_query(monkeypatch):
    calls = []
    moto = [{"vehicle_type": "motorcycle", "plan": "hourly", "price": 1.0}]
    monkeypatch.setattr(queries, "get_prices", lambda vehicle_type: calls.append(vehicle_type) or moto)

    result = tools.lookup_prices.invoke({"vehicle_type": "motorcycle"})

    assert json.loads(result) == moto
    assert calls == ["motorcycle"]


def test_lookup_prices_unknown_vehicle_type_lists_valid_types(monkeypatch):
    monkeypatch.setattr(queries, "get_prices", lambda vehicle_type: [])
    monkeypatch.setattr(queries, "get_vehicle_types", lambda: ["motorcycle", "car"])

    result = json.loads(tools.lookup_prices.invoke({"vehicle_type": "truck"}))

    assert "error" in result
    assert result["valid_vehicle_types"] == ["motorcycle", "car"]


def test_lookup_availability_without_level_returns_levels_and_total(monkeypatch):
    monkeypatch.setattr(queries, "get_availability", lambda: [{"level": 1, "total_spaces": 80, "available_spaces": 12}])
    monkeypatch.setattr(queries, "get_total_availability", lambda: {"total_spaces": 80, "available_spaces": 12})

    result = json.loads(tools.lookup_availability.invoke({}))

    assert result["total"] == {"total_spaces": 80, "available_spaces": 12}
    assert result["levels"] == [{"level": 1, "total_spaces": 80, "available_spaces": 12}]


def test_lookup_availability_with_unknown_level_returns_error(monkeypatch):
    monkeypatch.setattr(queries, "get_level_availability", lambda level: None)
    result = json.loads(tools.lookup_availability.invoke({"level": 99}))
    assert "error" in result
