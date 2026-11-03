import pytest

from parking_chatbot.recording.formatting import (
    LineFormatError,
    VehicleLineInput,
    format_group_lines,
    format_vehicle_line,
)


def _vehicle(car_number="AB123CD", pass_id="PK-7K3Q9X-01", vehicle_type="car"):
    return VehicleLineInput(car_number=car_number, pass_id=pass_id, vehicle_type=vehicle_type)


def _call(**overrides):
    kwargs = {
        "booking_id": "PK-7K3Q9X",
        "name": "Anna Smith",
        "period_start": "2026-11-20T09:00:00Z",
        "period_end": "2026-11-20T12:00:00Z",
        "approval_time": "2026-11-19T14:32:05Z",
        "vehicle": _vehicle(),
    }
    kwargs.update(overrides)
    return format_vehicle_line(**kwargs)


# --- format_vehicle_line: happy path -----------------------------------------------------------


def test_format_vehicle_line_produces_the_exact_expected_line():
    line = _call()
    assert line == (
        "Anna Smith | AB123CD | 2026-11-20T09:00:00Z/2026-11-20T12:00:00Z | "
        "2026-11-19T14:32:05Z | PK-7K3Q9X-01 | car"
    )


def test_format_vehicle_line_converts_a_non_utc_offset_to_utc():
    # 14:32:05+05:30 is 09:02:05Z
    line = _call(approval_time="2026-11-19T14:32:05+05:30")
    assert "2026-11-19T09:02:05Z" in line


@pytest.mark.parametrize("vehicle_type", ["motorcycle", "three_wheeler", "car", "van"])
def test_format_vehicle_line_accepts_every_allowed_vehicle_type(vehicle_type):
    line = _call(vehicle=_vehicle(vehicle_type=vehicle_type))
    assert line.endswith(vehicle_type)


# --- rejecting forbidden characters: refused, not escaped --------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "Anna|Smith",
        "Anna\nSmith",
        "Anna\rSmith",
        "Anna\tSmith",
        "Anna\x00Smith",
        "Anna Smith",
        "Anna Smith",
    ],
)
def test_format_vehicle_line_rejects_a_name_with_a_forbidden_character(bad_value):
    with pytest.raises(LineFormatError) as error:
        _call(name=bad_value)
    assert error.value.field == "name"


def test_format_vehicle_line_rejects_a_car_number_with_a_pipe():
    with pytest.raises(LineFormatError) as error:
        _call(vehicle=_vehicle(car_number="AB|123"))
    assert error.value.field == "car_number"


def test_format_vehicle_line_rejects_a_pass_id_with_a_control_character():
    with pytest.raises(LineFormatError) as error:
        _call(vehicle=_vehicle(pass_id="PK-7K3Q9X\x07-01"))
    assert error.value.field == "pass_id"


def test_line_format_error_never_includes_the_rejected_value_in_its_message():
    with pytest.raises(LineFormatError) as error:
        _call(name="super-secret-name|injected")
    assert "super-secret-name" not in str(error.value)
    assert "injected" not in str(error.value)


# --- vehicle type and dates ----------------------------------------------------------------------


def test_format_vehicle_line_rejects_an_invalid_vehicle_type():
    with pytest.raises(LineFormatError) as error:
        _call(vehicle=_vehicle(vehicle_type="truck"))
    assert error.value.field == "vehicle_type"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("period_start", "not-a-date"),
        ("period_end", "not-a-date"),
        ("approval_time", "not-a-date"),
    ],
)
def test_format_vehicle_line_rejects_an_unparseable_date(field, value):
    with pytest.raises(LineFormatError) as error:
        _call(**{field: value})
    assert error.value.field == field


def test_format_vehicle_line_rejects_a_date_with_no_utc_offset():
    with pytest.raises(LineFormatError) as error:
        _call(approval_time="2026-11-19T14:32:05")  # naive - no offset at all
    assert error.value.field == "approval_time"
    assert "UTC offset" in str(error.value)


def test_format_vehicle_line_rejects_an_end_before_the_start():
    with pytest.raises(LineFormatError) as error:
        _call(period_start="2026-11-20T12:00:00Z", period_end="2026-11-20T09:00:00Z")
    assert error.value.field == "period_end"


def test_format_vehicle_line_rejects_an_end_equal_to_the_start():
    with pytest.raises(LineFormatError) as error:
        _call(period_start="2026-11-20T09:00:00Z", period_end="2026-11-20T09:00:00Z")
    assert error.value.field == "period_end"


# --- booking ID / pass ID format and consistency ------------------------------------------------


@pytest.mark.parametrize("bad_booking_id", ["PK-7K3Q9", "pk-7k3q9x", "7K3Q9X", "PK-7K3Q9X-01", "PK-000000"])
def test_format_vehicle_line_rejects_a_malformed_booking_id(bad_booking_id):
    with pytest.raises(LineFormatError) as error:
        _call(booking_id=bad_booking_id)
    assert error.value.field == "booking_id"


@pytest.mark.parametrize("bad_pass_id", ["PK-7K3Q9X", "PK-7K3Q9X-1", "PK-7K3Q9X-001", "not-a-pass-id"])
def test_format_vehicle_line_rejects_a_malformed_pass_id(bad_pass_id):
    with pytest.raises(LineFormatError) as error:
        _call(vehicle=_vehicle(pass_id=bad_pass_id))
    assert error.value.field == "pass_id"


def test_format_vehicle_line_rejects_a_pass_id_from_a_different_booking():
    with pytest.raises(LineFormatError) as error:
        _call(booking_id="PK-7K3Q9X", vehicle=_vehicle(pass_id="PK-ZZZZZZ-01"))
    assert error.value.field == "pass_id"


# --- maximum lengths -------------------------------------------------------------------------------


def test_format_vehicle_line_accepts_a_name_at_the_maximum_length():
    name = "A" * 101
    assert _call(name=name).startswith(name)


def test_format_vehicle_line_rejects_a_name_over_the_maximum_length():
    with pytest.raises(LineFormatError) as error:
        _call(name="A" * 102)
    assert error.value.field == "name"


def test_format_vehicle_line_rejects_a_car_number_over_the_maximum_length():
    with pytest.raises(LineFormatError) as error:
        _call(vehicle=_vehicle(car_number="A" * 11))
    assert error.value.field == "car_number"


# --- format_group_lines: one line per vehicle, all-or-nothing ----------------------------------


def test_format_group_lines_produces_one_line_per_vehicle_sharing_booker_and_period():
    lines = format_group_lines(
        booking_id="PK-7K3Q9X",
        name="Anna Smith",
        period_start="2026-11-20T09:00:00Z",
        period_end="2026-11-20T12:00:00Z",
        approval_time="2026-11-19T14:32:05Z",
        vehicles=[
            _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
            _vehicle(car_number="BBB222", pass_id="PK-7K3Q9X-02", vehicle_type="van"),
        ],
    )
    assert len(lines) == 2
    assert all(line.startswith("Anna Smith | ") for line in lines)
    assert all("2026-11-20T09:00:00Z/2026-11-20T12:00:00Z" in line for line in lines)
    assert all("2026-11-19T14:32:05Z" in line for line in lines)
    assert lines[0].endswith("PK-7K3Q9X-01 | car")
    assert lines[1].endswith("PK-7K3Q9X-02 | van")


def test_format_group_lines_rejects_an_empty_group():
    with pytest.raises(LineFormatError) as error:
        format_group_lines(
            booking_id="PK-7K3Q9X",
            name="Anna Smith",
            period_start="2026-11-20T09:00:00Z",
            period_end="2026-11-20T12:00:00Z",
            approval_time="2026-11-19T14:32:05Z",
            vehicles=[],
        )
    assert error.value.field == "vehicles"


def test_format_group_lines_is_all_or_nothing_when_one_vehicle_is_invalid():
    with pytest.raises(LineFormatError):
        format_group_lines(
            booking_id="PK-7K3Q9X",
            name="Anna Smith",
            period_start="2026-11-20T09:00:00Z",
            period_end="2026-11-20T12:00:00Z",
            approval_time="2026-11-19T14:32:05Z",
            vehicles=[
                _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
                _vehicle(car_number="BBB222", pass_id="PK-7K3Q9X-02", vehicle_type="truck"),  # invalid
            ],
        )
    # No partial result is possible to observe: format_group_lines either returns
    # the full list or raises before returning anything at all.


def test_format_group_lines_rejects_a_vehicle_from_a_different_booking():
    with pytest.raises(LineFormatError) as error:
        format_group_lines(
            booking_id="PK-7K3Q9X",
            name="Anna Smith",
            period_start="2026-11-20T09:00:00Z",
            period_end="2026-11-20T12:00:00Z",
            approval_time="2026-11-19T14:32:05Z",
            vehicles=[
                _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
                _vehicle(car_number="BBB222", pass_id="PK-ZZZZZZ-02"),  # wrong booking
            ],
        )
    assert error.value.field == "pass_id"
