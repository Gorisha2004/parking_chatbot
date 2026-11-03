"""Tools the agent uses to collect, confirm and submit a (group) reservation.

Every tool works only on the current conversation's draft: none of them can read
saved bookings, so no conversation can reveal another person's reservation. The one
exception is check_booking_status, which can look up any booking ID given to it - but
it only ever returns that booking's bare status, never a name, plate or other detail.
"""
import json
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError

from parking_chatbot import config as settings
from parking_chatbot.admin.channel import get_admin_channel
from parking_chatbot.admin.models import ApprovalRequest, RequestVehicle
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.reservations import NotEnoughSpacesError, create_reservation, get_free_spaces
from parking_chatbot.reservation.ids import BOOKING_ID_LENGTH, BOOKING_ID_PREFIX, ID_ALPHABET
from parking_chatbot.reservation.models import REQUIRED_FIELDS, ReservationDraft, Vehicle

DETAIL_FIELDS = ["first_name", "last_name", "start", "end"]
BOOKING_ID_PATTERN = re.compile(rf"^{BOOKING_ID_PREFIX}-[{ID_ALPHABET}]{{{BOOKING_ID_LENGTH}}}$")

# One draft per conversation, keyed by thread_id (in memory, like the chat history).
_drafts: dict[str, ReservationDraft] = {}
_summary_shown: set[str] = set()
# The last booking ID mentioned in each conversation (submitted, or already looked
# up), so "is it approved yet?" works without repeating the ID every time.
_last_booking_id: dict[str, str] = {}
# Booking IDs submitted in each conversation that have not yet been reported back to
# the user as decided. Polled by the CLI in the background (see chatbot/cli.py) so the
# user hears about an administrator's decision without having to ask.
_submitted_booking_ids: dict[str, list[str]] = {}


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

    request = ApprovalRequest(
        booking_id=saved.booking_id,
        first_name=draft.first_name,
        last_name=draft.last_name,
        start=draft.start,
        end=draft.end,
        vehicles=[RequestVehicle(pass_id=p.pass_id, car_number=p.car_number, vehicle_type=p.vehicle_type) for p in saved.passes],
    )
    get_admin_channel().send_request(request)

    _last_booking_id[tid] = saved.booking_id
    _submitted_booking_ids.setdefault(tid, []).append(saved.booking_id)
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
    """Discard the NEW reservation currently being built (collected but not yet
    submitted), if the user changes their mind about it.

    This can NEVER cancel, change or otherwise affect a booking that has already
    been submitted - that is out of scope. Submitted bookings cannot be cancelled;
    use check_booking_status to report a submitted booking's status instead."""
    tid = _thread_id(config)
    had_draft = tid in _drafts
    _clear(tid)
    if had_draft:
        return json.dumps(
            {"status": "draft discarded", "message": "The in-progress reservation you were building has been discarded."}
        )
    return json.dumps(
        {
            "status": "nothing to discard",
            "message": "There was no in-progress reservation draft to discard. This does not "
            "affect any already-submitted booking - submitted bookings cannot be cancelled.",
        }
    )


@tool
def check_booking_status(config: RunnableConfig, booking_id: str | None = None) -> str:
    """Check the status of a reservation by its booking ID (for example PK-GC6ZMZ).
    If the user asks about status without giving an ID, omit this argument: the last
    booking ID mentioned in this conversation (for example one just submitted) is used
    automatically. If none is on record, this returns an error asking for one - ask the
    user for their booking ID rather than guessing it.
    Returns the status (pending, approved, refused or expired) and the administrator's
    reason if there is one. This never returns anyone's name, vehicle or other personal
    detail - only the bare status for the exact ID given, read-only."""
    tid = _thread_id(config)
    if booking_id is None:
        booking_id = _last_booking_id.get(tid)
        if booking_id is None:
            return json.dumps(
                {"error": "No booking ID was given, and none has come up yet in this conversation. Ask the user for their booking ID."}
            )
    else:
        booking_id = booking_id.strip().upper()
        if not BOOKING_ID_PATTERN.match(booking_id):
            return json.dumps({"error": f"'{booking_id}' is not a valid booking ID. It should look like PK-XXXXXX."})

    result = admin_db.get_booking(booking_id)
    if result is None:
        return json.dumps({"error": f"No booking found with ID {booking_id}."})

    _last_booking_id[tid] = booking_id
    return json.dumps(
        {
            "booking_id": result["booking_id"],
            "status": result["status"],
            "decision_reason": result["decision_reason"],
        }
    )


def get_draft(thread_id: str) -> ReservationDraft | None:
    """The reservation currently being collected in this conversation, or None.

    A read-only accessor, not an agent tool: the LangGraph orchestration layer
    (Stage 4) mirrors the draft into its typed shared state so the workflow can
    see what has been collected without reaching into this module's internals or
    re-implementing any of the collection logic that lives in the tools above.
    """
    return _drafts.get(thread_id)


def summary_was_shown(thread_id: str) -> bool:
    """Whether the user has already been shown a summary to confirm. Read-only
    accessor for the orchestration layer, for the same reason as get_draft()."""
    return thread_id in _summary_shown


def last_submitted_booking_id(thread_id: str) -> str | None:
    """The last booking ID this conversation submitted or looked up, or None.

    Read-only accessor for the orchestration layer: submit_reservation returns
    the new booking ID only inside the tool message the LLM sees, so this is how
    a graph node learns which booking a turn just created.
    """
    return _last_booking_id.get(thread_id)


def pending_decisions(thread_id: str) -> list[dict]:
    """Bookings submitted in this conversation that have just been decided (no longer
    pending). Not an agent tool - called by the CLI's background notifier, not the LLM.
    Each booking is reported once: it stops being tracked as soon as it is returned."""
    still_pending = []
    decided = []
    for booking_id in _submitted_booking_ids.get(thread_id, []):
        booking = admin_db.get_booking(booking_id)
        if booking is None or booking["status"] == "pending":
            still_pending.append(booking_id)
        else:
            decided.append(booking)
    _submitted_booking_ids[thread_id] = still_pending
    return decided


RESERVATION_TOOLS = [
    update_reservation_details,
    add_vehicles,
    remove_vehicle,
    show_reservation_summary,
    submit_reservation,
    cancel_reservation,
    check_booking_status,
]
