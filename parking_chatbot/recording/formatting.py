"""Pure rendering and validation for the Stage 3 reservation file's line format.

No file I/O here - this module only turns already-approved booking data into exact
text lines, or refuses to. Forbidden content is always rejected outright, never
escaped: an escaping scheme is one more place to get an edge case wrong, and
whatever later reads this file would have to un-escape it the same way.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from parking_chatbot.reservation.ids import BOOKING_ID_LENGTH, BOOKING_ID_PREFIX, ID_ALPHABET

VEHICLE_TYPES = ("motorcycle", "three_wheeler", "car", "van")

FIELD_SEPARATOR = " | "
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# first_name and last_name are each validated elsewhere at 2-50 characters; "name"
# here is the two joined with a space, so the worst case is 50 + 1 + 50.
MAX_NAME_LENGTH = 101
# Matches the upper bound of the car-number format validated at booking time
# (reservation/models.py's CAR_NUMBER_PATTERN).
MAX_CAR_NUMBER_LENGTH = 10

BOOKING_ID_PATTERN = re.compile(rf"^{BOOKING_ID_PREFIX}-[{ID_ALPHABET}]{{{BOOKING_ID_LENGTH}}}$")
PASS_ID_PATTERN = re.compile(rf"^{BOOKING_ID_PREFIX}-[{ID_ALPHABET}]{{{BOOKING_ID_LENGTH}}}-\d{{2}}$")

# Control characters (C0 range, plus DEL) and the Unicode line/paragraph separators -
# on top of the "|" field separator itself, checked separately below. All of these
# can make a line look like more than one line to whatever reads the file later.
_FORBIDDEN_CHAR_RANGES = ((0x00, 0x1F), (0x7F, 0x7F), (0x2028, 0x2028), (0x2029, 0x2029))


class LineFormatError(ValueError):
    """A field failed validation for the reservation file's line format.

    The message never includes the rejected value itself - only which field and
    why, as a category - because this text can end up in logs or error responses,
    and raw request content must never appear there (see STAGE_3.md's logging rule).
    """

    def __init__(self, field: str, reason: str):
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class VehicleLineInput:
    """One vehicle's share of the data needed for its line in the file."""

    car_number: str
    pass_id: str
    vehicle_type: str


def _check_no_forbidden_characters(field: str, value: str) -> None:
    if "|" in value:
        raise LineFormatError(field, "contains the '|' field separator")
    for character in value:
        code_point = ord(character)
        if any(low <= code_point <= high for low, high in _FORBIDDEN_CHAR_RANGES):
            raise LineFormatError(field, "contains a newline, control character, or line/paragraph separator")


def _check_max_length(field: str, value: str, max_length: int) -> None:
    if len(value) > max_length:
        raise LineFormatError(field, f"is longer than the maximum of {max_length} characters")


def _check_vehicle_type(vehicle_type: str) -> None:
    if vehicle_type not in VEHICLE_TYPES:
        raise LineFormatError("vehicle_type", f"must be one of: {', '.join(VEHICLE_TYPES)}")


def _check_booking_id(booking_id: str) -> None:
    if not BOOKING_ID_PATTERN.match(booking_id):
        raise LineFormatError("booking_id", f"is not a valid booking ID (expected {BOOKING_ID_PREFIX}-XXXXXX)")


def _check_pass_id(booking_id: str, pass_id: str) -> None:
    if not PASS_ID_PATTERN.match(pass_id):
        raise LineFormatError("pass_id", f"is not a valid pass ID (expected {BOOKING_ID_PREFIX}-XXXXXX-NN)")
    if not pass_id.startswith(f"{booking_id}-"):
        raise LineFormatError("pass_id", "does not belong to the given booking ID")


def _parse_utc(field: str, value: str) -> datetime:
    """Parse an ISO 8601 date-time string and convert it to UTC. Rejects anything
    that doesn't parse, and anything with no UTC offset at all - an instant in time,
    as this file records, must be unambiguous."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise LineFormatError(field, "is not a valid ISO 8601 date-time") from None
    if parsed.tzinfo is None:
        raise LineFormatError(field, "is not a valid ISO 8601 date-time (missing a UTC offset)")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.strftime(_DATE_FORMAT)


def format_vehicle_line(
    *,
    booking_id: str,
    name: str,
    period_start: str,
    period_end: str,
    approval_time: str,
    vehicle: VehicleLineInput,
) -> str:
    """Render the exact line for one vehicle:
    `Name | Car Number | Reservation Period | Approval Time | Pass ID | Vehicle Type`

    Raises LineFormatError for the first field that fails validation; nothing is
    ever escaped or altered - a field that isn't safe to write is refused outright.
    """
    _check_booking_id(booking_id)
    _check_pass_id(booking_id, vehicle.pass_id)

    for field, value in (("name", name), ("car_number", vehicle.car_number)):
        _check_no_forbidden_characters(field, value)
    _check_max_length("name", name, MAX_NAME_LENGTH)
    _check_max_length("car_number", vehicle.car_number, MAX_CAR_NUMBER_LENGTH)
    _check_vehicle_type(vehicle.vehicle_type)

    start = _parse_utc("period_start", period_start)
    end = _parse_utc("period_end", period_end)
    if end <= start:
        raise LineFormatError("period_end", "must be after period_start")
    approved = _parse_utc("approval_time", approval_time)

    period = f"{_format_utc(start)}/{_format_utc(end)}"
    fields = [name, vehicle.car_number, period, _format_utc(approved), vehicle.pass_id, vehicle.vehicle_type]
    return FIELD_SEPARATOR.join(fields)


def format_group_lines(
    *,
    booking_id: str,
    name: str,
    period_start: str,
    period_end: str,
    approval_time: str,
    vehicles: list[VehicleLineInput],
) -> list[str]:
    """Render one line per vehicle in a group booking, all sharing one booking ID.

    All-or-nothing: if any vehicle's data fails validation, the exception
    propagates immediately and no line for this group is returned at all - a
    partial group must never reach the file.
    """
    if not vehicles:
        raise LineFormatError("vehicles", "a booking must have at least one vehicle")
    return [
        format_vehicle_line(
            booking_id=booking_id,
            name=name,
            period_start=period_start,
            period_end=period_end,
            approval_time=approval_time,
            vehicle=vehicle,
        )
        for vehicle in vehicles
    ]
