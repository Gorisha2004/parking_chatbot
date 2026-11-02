from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from parking_chatbot import config
from parking_chatbot.reservation.models import VEHICLE_TYPES, ReservationDraft, Vehicle


def _vehicles(count: int) -> list[Vehicle]:
    return [Vehicle(car_number=f"CAR{i:03d}", vehicle_type="car") for i in range(1, count + 1)]


# --- Vehicle -------------------------------------------------------------------------------------


@pytest.mark.parametrize("vehicle_type", VEHICLE_TYPES)
def test_accepts_every_known_vehicle_type(vehicle_type):
    assert Vehicle(car_number="AB123CD", vehicle_type=vehicle_type).vehicle_type == vehicle_type


@pytest.mark.parametrize("raw", ["Three Wheeler", "three-wheeler", "  THREE_WHEELER "])
def test_vehicle_type_is_normalised(raw):
    assert Vehicle(car_number="AB123CD", vehicle_type=raw).vehicle_type == "three_wheeler"


@pytest.mark.parametrize("bad", ["truck", "bus", "spaceship", ""])
def test_rejects_unknown_vehicle_type_and_lists_valid_ones(bad):
    with pytest.raises(ValidationError, match="motorcycle, three_wheeler, car, van"):
        Vehicle(car_number="AB123CD", vehicle_type=bad)


def test_car_number_is_normalised_upper_and_stripped_of_separators():
    assert Vehicle(car_number="ab-123 cd", vehicle_type="car").car_number == "AB123CD"


@pytest.mark.parametrize("bad", ["!!", "AB1", "ABCDEFGHIJK", "AB|12345", "AB/../12"])
def test_rejects_invalid_car_numbers_including_separator_injection(bad):
    with pytest.raises(ValidationError):
        Vehicle(car_number=bad, vehicle_type="car")


def test_newlines_in_a_car_number_are_stripped_never_stored():
    assert Vehicle(car_number="AB12\n34", vehicle_type="car").car_number == "AB1234"


# --- Booker and period ---------------------------------------------------------------------------


def test_missing_fields_lists_every_unset_required_field():
    draft = ReservationDraft(first_name="Anna")
    assert draft.missing_fields() == ["last_name", "vehicles", "start", "end"]
    assert draft.is_complete() is False


def test_names_are_title_cased_and_stripped():
    draft = ReservationDraft(first_name="  anna ", last_name="O'Brien")
    assert draft.first_name == "Anna"
    assert draft.last_name == "O'Brien"


@pytest.mark.parametrize("bad_name", ["A", "Anna1", "", "   ", "Anna|Smith"])
def test_rejects_invalid_names(bad_name):
    with pytest.raises(ValidationError):
        ReservationDraft(first_name=bad_name)


def test_rejects_reservation_starting_in_the_past():
    with pytest.raises(ValidationError):
        ReservationDraft(start=datetime(2020, 1, 1, 10, 0))


def test_rejects_end_before_or_equal_to_start():
    start = datetime.now() + timedelta(days=1)
    with pytest.raises(ValidationError):
        ReservationDraft(start=start, end=start)


def test_rejects_reservation_longer_than_max_days():
    start = datetime.now() + timedelta(days=1)
    with pytest.raises(ValidationError):
        ReservationDraft(start=start, end=start + timedelta(days=31))


def test_accepts_start_just_inside_the_advance_booking_window():
    start = datetime.now() + timedelta(days=config.MAX_ADVANCE_DAYS - 1)
    draft = ReservationDraft(start=start, end=start + timedelta(hours=2))
    assert draft.start == start


@pytest.mark.parametrize("days_over_limit", [1, 640], ids=["one day too far", "about 2 years ahead"])
def test_rejects_start_beyond_the_advance_booking_window(days_over_limit):
    with pytest.raises(ValidationError, match="in advance"):
        ReservationDraft(start=datetime.now() + timedelta(days=config.MAX_ADVANCE_DAYS + days_over_limit))


def test_advance_booking_window_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "MAX_ADVANCE_DAYS", 7)
    with pytest.raises(ValidationError, match="7 days in advance"):
        ReservationDraft(start=datetime.now() + timedelta(days=8))


def test_rejects_end_alone_far_in_the_future():
    with pytest.raises(ValidationError, match="too far in the future"):
        ReservationDraft(end=datetime(2028, 6, 1, 12, 0))


def test_rejects_end_alone_in_the_past():
    with pytest.raises(ValidationError, match="cannot end in the past"):
        ReservationDraft(end=datetime(2020, 1, 1, 12, 0))


# --- Group of vehicles ---------------------------------------------------------------------------


def test_accepts_a_group_up_to_the_configured_maximum():
    draft = ReservationDraft(vehicles=_vehicles(config.MAX_VEHICLES_PER_BOOKING))
    assert len(draft.vehicles) == config.MAX_VEHICLES_PER_BOOKING


def test_rejects_a_group_above_the_configured_maximum():
    with pytest.raises(ValidationError, match=f"at most {config.MAX_VEHICLES_PER_BOOKING} vehicles"):
        ReservationDraft(vehicles=_vehicles(config.MAX_VEHICLES_PER_BOOKING + 1))


def test_group_size_limit_comes_from_config(monkeypatch):
    monkeypatch.setattr(config, "MAX_VEHICLES_PER_BOOKING", 2)
    with pytest.raises(ValidationError, match="at most 2 vehicles"):
        ReservationDraft(vehicles=_vehicles(3))


def test_rejects_the_same_car_number_twice_in_one_booking():
    twice = [Vehicle(car_number="AB123CD", vehicle_type="car"), Vehicle(car_number="ab-123 cd", vehicle_type="van")]
    with pytest.raises(ValidationError, match="only appear once"):
        ReservationDraft(vehicles=twice)


def test_with_vehicle_adds_new_vehicles_in_order():
    draft = ReservationDraft().with_vehicle(Vehicle(car_number="AAA111", vehicle_type="car"))
    draft = draft.with_vehicle(Vehicle(car_number="BBB222", vehicle_type="van"))
    assert [v.car_number for v in draft.vehicles] == ["AAA111", "BBB222"]


def test_with_vehicle_updates_the_type_of_a_plate_already_in_the_booking():
    draft = ReservationDraft(vehicles=_vehicles(2))
    draft = draft.with_vehicle(Vehicle(car_number="car001", vehicle_type="van"))
    assert [(v.car_number, v.vehicle_type) for v in draft.vehicles] == [("CAR001", "van"), ("CAR002", "car")]


def test_with_vehicle_refuses_to_exceed_the_maximum(monkeypatch):
    monkeypatch.setattr(config, "MAX_VEHICLES_PER_BOOKING", 2)
    draft = ReservationDraft(vehicles=_vehicles(2))
    with pytest.raises(ValidationError):
        draft.with_vehicle(Vehicle(car_number="NEW999", vehicle_type="car"))


def test_without_vehicle_removes_by_normalised_plate():
    draft = ReservationDraft(vehicles=_vehicles(2)).without_vehicle("car-001")
    assert [v.car_number for v in draft.vehicles] == ["CAR002"]


def test_is_complete_true_when_all_required_fields_set():
    start = datetime.now() + timedelta(days=1)
    draft = ReservationDraft(
        first_name="Anna",
        last_name="Smith",
        vehicles=_vehicles(3),
        start=start,
        end=start + timedelta(hours=1),
    )
    assert draft.is_complete() is True
    assert draft.missing_fields() == []
