"""The reservation domain model: one booker and one period for one or more vehicles."""
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, model_validator

from parking_chatbot import config

NAME_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ '\-]{1,49}$")
CAR_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")
VEHICLE_TYPES = ("motorcycle", "three_wheeler", "car", "van")
REQUIRED_FIELDS = ["first_name", "last_name", "vehicles", "start", "end"]
MAX_DAYS = 30


def normalize_car_number(value: str) -> str:
    """Upper-case the plate and drop spaces and hyphens, so 'ab-12 cd' == 'AB12CD'."""
    return re.sub(r"[\s\-]", "", value).upper()


class Vehicle(BaseModel):
    """One vehicle in a booking."""

    car_number: str
    vehicle_type: str

    @field_validator("car_number")
    @classmethod
    def validate_car_number(cls, value: str) -> str:
        cleaned = normalize_car_number(value)
        if not CAR_NUMBER_PATTERN.match(cleaned):
            raise ValueError("Car number must be 4-10 letters or digits (spaces and hyphens are ignored).")
        return cleaned

    @field_validator("vehicle_type")
    @classmethod
    def validate_vehicle_type(cls, value: str) -> str:
        cleaned = re.sub(r"[\s\-]+", "_", value.strip().lower())
        if cleaned not in VEHICLE_TYPES:
            raise ValueError(f"Vehicle type must be one of: {', '.join(VEHICLE_TYPES)}.")
        return cleaned


class ReservationDraft(BaseModel):
    """A reservation being collected: every field starts empty and is validated when set."""

    first_name: str | None = None
    last_name: str | None = None
    vehicles: list[Vehicle] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not NAME_PATTERN.match(value):
            raise ValueError("Names must be 2-50 letters (spaces, hyphens, apostrophes allowed).")
        return value.title()

    @field_validator("vehicles")
    @classmethod
    def validate_vehicles(cls, vehicles: list[Vehicle]) -> list[Vehicle]:
        if len(vehicles) > config.MAX_VEHICLES_PER_BOOKING:
            raise ValueError(f"A booking can include at most {config.MAX_VEHICLES_PER_BOOKING} vehicles.")
        plates = [vehicle.car_number for vehicle in vehicles]
        if len(set(plates)) != len(plates):
            raise ValueError("Each car number can only appear once in a booking.")
        return vehicles

    @field_validator("start", "end")
    @classmethod
    def drop_timezone(cls, value: datetime | None) -> datetime | None:
        return value.replace(tzinfo=None) if value else value

    @model_validator(mode="after")
    def validate_period(self) -> "ReservationDraft":
        now = datetime.now()
        latest_start = now + timedelta(days=config.MAX_ADVANCE_DAYS)
        if self.start:
            if self.start < now:
                raise ValueError("The reservation cannot start in the past.")
            if self.start > latest_start:
                raise ValueError(
                    f"Reservations can only start up to {config.MAX_ADVANCE_DAYS} days in advance "
                    f"(latest start: {latest_start:%Y-%m-%d})."
                )
        if self.end:
            if self.end < now:
                raise ValueError("The reservation cannot end in the past.")
            if self.end > latest_start + timedelta(days=MAX_DAYS):
                raise ValueError(
                    f"The end date is too far in the future: reservations can only start up to "
                    f"{config.MAX_ADVANCE_DAYS} days in advance and last at most {MAX_DAYS} days."
                )
        if self.start and self.end:
            if self.end <= self.start:
                raise ValueError("The end of the reservation must be after its start.")
            if (self.end - self.start).days > MAX_DAYS:
                raise ValueError(f"A reservation cannot be longer than {MAX_DAYS} days.")
        return self

    def with_vehicle(self, vehicle: Vehicle) -> "ReservationDraft":
        """A copy with the vehicle added, or with its type updated if the plate is already booked."""
        vehicles = [vehicle if v.car_number == vehicle.car_number else v for v in self.vehicles]
        if vehicle.car_number not in {v.car_number for v in self.vehicles}:
            vehicles.append(vehicle)
        return ReservationDraft(**{**self.model_dump(), "vehicles": vehicles})

    def without_vehicle(self, car_number: str) -> "ReservationDraft":
        """A copy without the vehicle with this plate (unchanged if it is not in the booking)."""
        plate = normalize_car_number(car_number)
        vehicles = [v for v in self.vehicles if v.car_number != plate]
        return ReservationDraft(**{**self.model_dump(), "vehicles": vehicles})

    def missing_fields(self) -> list[str]:
        return [name for name in REQUIRED_FIELDS if getattr(self, name) in (None, [])]

    def is_complete(self) -> bool:
        return not self.missing_fields()
