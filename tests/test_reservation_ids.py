import re

import pytest

from parking_chatbot.reservation import ids

BOOKING_ID = re.compile(r"^PK-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$")


def test_alphabet_has_no_look_alike_characters():
    assert not set("0O1IL") & set(ids.ID_ALPHABET)


def test_new_booking_id_has_the_expected_format():
    for _ in range(50):
        assert BOOKING_ID.match(ids.new_booking_id())


def test_new_booking_ids_are_random_not_sequential():
    generated = {ids.new_booking_id() for _ in range(200)}
    assert len(generated) == 200


def test_new_booking_id_uses_the_secrets_module(monkeypatch):
    monkeypatch.setattr(ids.secrets, "choice", lambda alphabet: "K")
    assert ids.new_booking_id() == "PK-KKKKKK"


@pytest.mark.parametrize(("number", "expected"), [(1, "PK-7K3Q9X-01"), (3, "PK-7K3Q9X-03"), (99, "PK-7K3Q9X-99")])
def test_pass_id_appends_a_two_digit_vehicle_number(number, expected):
    assert ids.pass_id("PK-7K3Q9X", number) == expected


@pytest.mark.parametrize("number", [0, 100, -1])
def test_pass_id_rejects_out_of_range_vehicle_numbers(number):
    with pytest.raises(ValueError):
        ids.pass_id("PK-7K3Q9X", number)
