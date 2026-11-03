import stat
import threading

import pytest

from parking_chatbot import config
from parking_chatbot.recording.formatting import LineFormatError, VehicleLineInput
from parking_chatbot.recording import writer


@pytest.fixture(autouse=True)
def reservations_file(tmp_path, monkeypatch):
    """Point MCP_RESERVATIONS_FILE_PATH at a fresh temp file for every test - the
    equivalent, for this file, of the sql_db fixture for the database."""
    path = tmp_path / "reservations.txt"
    monkeypatch.setattr(config, "MCP_RESERVATIONS_FILE_PATH", path)
    return path


def _vehicle(car_number="AB123CD", pass_id="PK-7K3Q9X-01", vehicle_type="car"):
    return VehicleLineInput(car_number=car_number, pass_id=pass_id, vehicle_type=vehicle_type)


def _record(**overrides):
    kwargs = {
        "booking_id": "PK-7K3Q9X",
        "name": "Anna Smith",
        "period_start": "2026-11-20T09:00:00Z",
        "period_end": "2026-11-20T12:00:00Z",
        "approval_time": "2026-11-19T14:32:05Z",
        "vehicles": [_vehicle()],
    }
    kwargs.update(overrides)
    return writer.record_group(**kwargs)


# --- happy path ------------------------------------------------------------------------------------


def test_record_group_writes_the_expected_line(reservations_file):
    result = _record()

    assert result == {"recorded": True, "lines_written": 1}
    assert reservations_file.read_text(encoding="utf-8") == (
        "Anna Smith | AB123CD | 2026-11-20T09:00:00Z/2026-11-20T12:00:00Z | "
        "2026-11-19T14:32:05Z | PK-7K3Q9X-01 | car\n"
    )


def test_record_group_writes_one_line_per_vehicle(reservations_file):
    result = _record(
        vehicles=[
            _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
            _vehicle(car_number="BBB222", pass_id="PK-7K3Q9X-02", vehicle_type="van"),
        ]
    )

    assert result == {"recorded": True, "lines_written": 2}
    lines = reservations_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("Anna Smith | AAA111")
    assert lines[1].startswith("Anna Smith | BBB222")


def test_record_group_appends_without_disturbing_a_prior_booking(reservations_file):
    _record(booking_id="PK-AAAAAA", vehicles=[_vehicle(pass_id="PK-AAAAAA-01")])
    _record(booking_id="PK-BBBBBB", vehicles=[_vehicle(pass_id="PK-BBBBBB-01")])

    lines = reservations_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "PK-AAAAAA-01" in lines[0]
    assert "PK-BBBBBB-01" in lines[1]


def test_record_group_creates_the_file_with_owner_only_permissions(reservations_file, monkeypatch):
    calls = []
    monkeypatch.setattr(writer.os, "chmod", lambda path, mode: calls.append(mode))

    _record()

    assert calls == [stat.S_IRUSR | stat.S_IWUSR]


def test_record_group_does_not_re_apply_permissions_on_a_later_call(reservations_file, monkeypatch):
    calls = []
    monkeypatch.setattr(writer.os, "chmod", lambda path, mode: calls.append(mode))

    _record(booking_id="PK-AAAAAA", vehicles=[_vehicle(pass_id="PK-AAAAAA-01")])
    _record(booking_id="PK-BBBBBB", vehicles=[_vehicle(pass_id="PK-BBBBBB-01")])

    assert len(calls) == 1  # only when the file was first created


# --- dedup by booking ID --------------------------------------------------------------------------


def test_record_group_is_a_no_op_when_the_booking_id_is_already_recorded(reservations_file):
    _record()
    first_contents = reservations_file.read_text(encoding="utf-8")

    result = _record()  # e.g. a retried send after the server's own response was lost

    assert result == {"recorded": True, "lines_written": 0}
    assert reservations_file.read_text(encoding="utf-8") == first_contents  # unchanged, not duplicated


def test_record_group_dedups_by_booking_id_even_with_different_vehicle_data(reservations_file):
    """The dedup check only looks at the booking ID's pass-ID prefix - a retry
    that (incorrectly) carried different data still must not write a second time."""
    _record()

    result = _record(vehicles=[_vehicle(car_number="ZZZ999", pass_id="PK-7K3Q9X-01")])

    assert result["lines_written"] == 0
    assert "ZZZ999" not in reservations_file.read_text(encoding="utf-8")


# --- all-or-nothing: an invalid group writes nothing ----------------------------------------------


def test_record_group_with_invalid_data_writes_nothing(reservations_file):
    with pytest.raises(LineFormatError):
        _record(vehicles=[_vehicle(vehicle_type="truck")])

    assert not reservations_file.exists()


def test_record_group_with_one_invalid_vehicle_in_a_group_writes_none_of_it(reservations_file):
    with pytest.raises(LineFormatError):
        _record(
            vehicles=[
                _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
                _vehicle(car_number="BBB222", pass_id="PK-7K3Q9X-02", vehicle_type="truck"),  # invalid
            ]
        )

    assert not reservations_file.exists()


def test_record_group_writes_a_whole_group_in_a_single_write_call(reservations_file, monkeypatch):
    """STAGE_3.md requires the whole group written in one append - checked here by
    counting the actual write() calls on the opened file, not just the end result."""
    from pathlib import Path

    real_path_open = Path.open
    writes = []

    class _CountingFile:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._handle.close()

        def write(self, data):
            writes.append(data)
            return self._handle.write(data)

    def fake_path_open(self, mode="r", encoding=None):
        handle = real_path_open(self, mode, encoding=encoding)
        return _CountingFile(handle) if mode == "a" else handle

    monkeypatch.setattr(Path, "open", fake_path_open)

    _record(
        vehicles=[
            _vehicle(car_number="AAA111", pass_id="PK-7K3Q9X-01"),
            _vehicle(car_number="BBB222", pass_id="PK-7K3Q9X-02", vehicle_type="van"),
        ]
    )

    assert len(writes) == 1
    assert writes[0].count("\n") == 2  # both lines, in that one write


# --- recovering from a crash-truncated last line --------------------------------------------------


def test_record_group_does_not_merge_onto_a_truncated_last_line(reservations_file):
    """A process killed mid-write() can leave a half-written last line with no
    trailing newline. The next append must not silently glue new data onto it."""
    reservations_file.write_text("Anna Smith | AB123CD | 2026-11-20T09:00:00Z/2026-1", encoding="utf-8")

    _record(booking_id="PK-BBBBBB", vehicles=[_vehicle(car_number="BBB222", pass_id="PK-BBBBBB-01")])

    lines = reservations_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1] == (
        "Anna Smith | BBB222 | 2026-11-20T09:00:00Z/2026-11-20T12:00:00Z | "
        "2026-11-19T14:32:05Z | PK-BBBBBB-01 | car"
    )


def test_record_group_does_not_add_a_blank_line_after_a_clean_last_line(reservations_file):
    _record(booking_id="PK-AAAAAA", vehicles=[_vehicle(pass_id="PK-AAAAAA-01")])

    _record(booking_id="PK-BBBBBB", vehicles=[_vehicle(pass_id="PK-BBBBBB-01")])

    contents = reservations_file.read_text(encoding="utf-8")
    assert "\n\n" not in contents


# --- concurrency -----------------------------------------------------------------------------------


def test_two_concurrent_calls_for_the_same_booking_write_it_only_once(reservations_file):
    errors = []

    def call():
        try:
            _record()
        except Exception as error:  # the test only cares that neither call raises or corrupts the file
            errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    lines = reservations_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_two_concurrent_calls_for_different_bookings_both_succeed_without_corruption(reservations_file):
    errors = []

    def call(booking_id, car_number):
        try:
            _record(booking_id=booking_id, vehicles=[_vehicle(car_number=car_number, pass_id=f"{booking_id}-01")])
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=call, args=("PK-AAAAAA", "AAA111")),
        threading.Thread(target=call, args=("PK-BBBBBB", "BBB222")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    lines = reservations_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # neither line is missing, truncated or merged with the other
    assert any("AAA111" in line for line in lines)
    assert any("BBB222" in line for line in lines)
