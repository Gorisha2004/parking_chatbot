"""Tools the agent uses to collect, confirm and submit a (group) reservation.

Every tool works only on the current conversation's draft: none of them can read
saved bookings, so no conversation can reveal another person's reservation.
"""
import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError

from parking_chatbot import config as settings
from parking_chatbot.db.reservations import NotEnoughSpacesError, create_reservation, get_free_spaces
from parking_chatbot.reservation.models import REQUIRED_FIELDS, ReservationDraft, Vehicle

DETAIL_FIELDS = ["first_name", "last_name", "start", "end"]

# One draft per conversation, keyed by thread_id (in memory, like the chat history).
_drafts: dict[str, ReservationDraft] = {}
_summary_shown: set[str] = set()


class VehicleInput(BaseModel):
    """A vehicle as the user described it; it is validated when added to the booking."""

    car_number: str
    vehicle_type: str


def _thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _clear(thread_id: str) -> None:
    _drafts.pop(thread_id, None)
    _summary_shown.discard(thread_id)


def _first_message(error: ValidationError) -> str:
    return error.errors()[0]["msg"].removeprefix("Value error, ")


def _save(thread_id: str, draft: ReservationDraft) -> dict:
    """Store the draft; any change means the user must see and confirm a new summary."""
    _drafts[thread_id] = draft
    _summary_shown.discard(thread_id)
    return {"still_missing": draft.missing_fields(), "current_details": draft.model_dump(mode="json")}


def _not_enough_spaces(free: int, needed: int) -> str:
    return json.dumps(
        {
            "error": "Not enough free spaces for this period",
            "free_spaces": free,
            "vehicles_in_booking": needed,
            "next_step": "Suggest fewer vehicles or a different period.",
        }
    )


@tool
def update_reservation_details(
    config: RunnableConfig,
    first_name: str | None = None,
    last_name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Save the booker's name and the reservation period the user has just provided.
    One booker and one period apply to every vehicle in the booking. Pass ONLY the fields
    the user actually gave, never guess or invent values. Start and end must be local
    times in the format YYYY-MM-DDTHH:MM; work out relative dates such as 'tomorrow'
    from the current date in your instructions. Returns what was saved, which fields were
    rejected and why, and which fields are still missing."""
    tid = _thread_id(config)
    draft = _drafts.get(tid, ReservationDraft())
    provided = {"first_name": first_name, "last_name": last_name, "start": start, "end": end}

    saved, errors = [], {}
    for field in DETAIL_FIELDS:
        if provided[field] is None:
            continue
        try:
            draft = ReservationDraft(**{**draft.model_dump(), field: provided[field]})
            saved.append(field)
        except ValidationError as error:
            errors[field] = _first_message(error)

    return json.dumps({"saved": saved, "rejected": errors, **_save(tid, draft)})


@tool
def add_vehicles(config: RunnableConfig, vehicles: list[VehicleInput]) -> str:
    """Add one or more vehicles to the reservation. A booking can be for one vehicle or a
    group (for example a family gathering): every vehicle shares the same booker and period.
    Pass all the vehicles the user listed in one call. For each one give the car number and
    the vehicle type, one of: motorcycle (motorcycles, scooters, two-wheelers),
    three_wheeler (auto-rickshaws), car (cars, SUVs) or van (small vans, minibuses).
    Adding a car number that is already in the booking updates its vehicle type.
    Returns which vehicles were added, which were rejected and why, and the group limit."""
    tid = _thread_id(config)
    draft = _drafts.get(tid, ReservationDraft())

    added, errors = [], {}
    for item in vehicles:
        try:
            vehicle = Vehicle(car_number=item.car_number, vehicle_type=item.vehicle_type)
            draft = draft.with_vehicle(vehicle)
            added.append(vehicle.car_number)
        except ValidationError as error:
            errors[item.car_number] = _first_message(error)

    return json.dumps(
        {
            "added": added,
            "rejected": errors,
            "vehicle_count": len(draft.vehicles),
            "max_vehicles": settings.MAX_VEHICLES_PER_BOOKING,
            **_save(tid, draft),
        }
    )


@tool
def remove_vehicle(config: RunnableConfig, car_number: str) -> str:
    """Remove a vehicle from the reservation by its car number, if the user no longer
    wants it in the booking."""
    tid = _thread_id(config)
    draft = _drafts.get(tid, ReservationDraft())
    updated = draft.without_vehicle(car_number)
    removed = len(updated.vehicles) < len(draft.vehicles)
    return json.dumps({"removed": removed, "vehicle_count": len(updated.vehicles), **_save(tid, updated)})


@tool
def show_reservation_summary(config: RunnableConfig) -> str:
    """Return the completed reservation (booker, period and every vehicle) so it can be read
    back to the user for confirmation. Also checks that enough spaces are free for the whole
    group in that period. Call this when nothing is missing, before submitting."""
    tid = _thread_id(config)
    draft = _drafts.get(tid)
    if draft is None or not draft.is_complete():
        missing = draft.missing_fields() if draft else REQUIRED_FIELDS
        return json.dumps({"error": "The reservation is not complete yet", "still_missing": missing})

    free = get_free_spaces(draft.start, draft.end)
    if free < len(draft.vehicles):
        return _not_enough_spaces(free, len(draft.vehicles))

    _summary_shown.add(tid)
    return json.dumps(
        {
            "details": draft.model_dump(mode="json"),
            "vehicle_count": len(draft.vehicles),
            "next_step": "Show these details to the user and ask them to confirm with yes or no. "
            "Do NOT submit until the user has explicitly confirmed.",
        }
    )


@tool
def submit_reservation(config: RunnableConfig) -> str:
    """Submit the reservation for administrator approval (the whole group is one request).
    ONLY call this after the user has seen the summary and explicitly confirmed with a clear
    yes. Returns the booking ID and one pass ID per vehicle."""
    tid = _thread_id(config)
    draft = _drafts.get(tid)
    if draft is None or not draft.is_complete():
        return json.dumps({"error": "The reservation is not complete yet"})
    if tid not in _summary_shown:
        return json.dumps({"error": "Call show_reservation_summary and get the user's confirmation first"})

    try:
        saved = create_reservation(draft)
    except NotEnoughSpacesError as error:
        _summary_shown.discard(tid)
        return _not_enough_spaces(error.free, error.needed)

    _clear(tid)
    return json.dumps(
        {
            "booking_id": saved.booking_id,
            "passes": [
                {"pass_id": p.pass_id, "car_number": p.car_number, "vehicle_type": p.vehicle_type}
                for p in saved.passes
            ],
            "status": "pending",
            "message": "Request saved and pending administrator approval; it is NOT confirmed yet. "
            "An administrator will review the whole booking. Each driver shows their own pass ID on arrival.",
        }
    )


@tool
def cancel_reservation(config: RunnableConfig) -> str:
    """Discard the reservation being collected, if the user changes their mind."""
    _clear(_thread_id(config))
    return json.dumps({"status": "reservation draft discarded"})


RESERVATION_TOOLS = [
    update_reservation_details,
    add_vehicles,
    remove_vehicle,
    show_reservation_summary,
    submit_reservation,
    cancel_reservation,
]
