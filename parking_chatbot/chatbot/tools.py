import json

from langchain_core.tools import tool

from parking_chatbot.db import queries
from parking_chatbot.rag.chain import format_context, retrieve
from parking_chatbot.chatbot.reservation_tools import RESERVATION_TOOLS


@tool
def search_parking_info(question: str) -> str:
    """Search the parking facility's documents for general, rarely-changing information:
    location and directions, facilities and services (EV charging, accessibility, security),
    which vehicles are allowed (vehicle types, height limit, gas-powered vehicles),
    parking rules, payment methods, contact details, FAQs and the booking process.
    Do NOT use this for opening hours, prices or free spaces.
    If the user asks about several different things in one message, call this
    tool separately for each one, using a short, focused query for each part."""
    return format_context(retrieve(question))


@tool
def lookup_working_hours() -> str:
    """Get the current opening and closing times for every day of the week."""
    return json.dumps(queries.get_working_hours())


@tool
def lookup_prices(vehicle_type: str | None = None) -> str:
    """Get the current parking prices (hourly, daily, weekly, monthly). Prices depend on
    the vehicle type, because bigger vehicles take more space:
    - motorcycle: motorcycles, scooters and other two-wheelers
    - three_wheeler: auto-rickshaws and other three-wheelers
    - car: cars and SUVs, including electric cars
    - van: small vans and minibuses up to 2.1 m tall
    Pass one of these vehicle types, or leave it empty to get the prices for all of them."""
    prices = queries.get_prices(vehicle_type)
    if vehicle_type is not None and not prices:
        return json.dumps(
            {"error": f"Unknown vehicle type '{vehicle_type}'", "valid_vehicle_types": queries.get_vehicle_types()}
        )
    return json.dumps(prices)


@tool
def lookup_availability(level: int | None = None) -> str:
    """Get the number of free parking spaces right now.
    Pass a level number (1-4) for one level, or leave it empty for all levels plus the total."""
    if level is None:
        return json.dumps(
            {
                "levels": queries.get_availability(),
                "total": queries.get_total_availability(),
            }
        )
    result = queries.get_level_availability(level)
    return json.dumps(result if result else {"error": f"Level {level} does not exist"})


ALL_TOOLS = [
    search_parking_info,
    lookup_working_hours,
    lookup_prices,
    lookup_availability,
    *RESERVATION_TOOLS,
]


if __name__ == "__main__":
    for t in ALL_TOOLS:
        print(t.name, "->", t.args)
    print(lookup_availability.invoke({"level": 3}))
    print(lookup_availability.invoke({"level": 99}))
    print(lookup_prices.invoke({}))
    print(search_parking_info.invoke({"question": "is there a petrol pump?"}))