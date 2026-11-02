"""Booking and pass IDs: random (unguessable), short, and easy to read out loud."""
import secrets

# Digits and capitals without the look-alikes 0/O and 1/I/L.
ID_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
BOOKING_ID_PREFIX = "PK"
BOOKING_ID_LENGTH = 6


def new_booking_id() -> str:
    """A fresh random booking ID such as PK-7K3Q9X. Uniqueness is enforced by the database."""
    code = "".join(secrets.choice(ID_ALPHABET) for _ in range(BOOKING_ID_LENGTH))
    return f"{BOOKING_ID_PREFIX}-{code}"


def pass_id(booking_id: str, vehicle_number: int) -> str:
    """The pass ID of the n-th vehicle (1-based) in a booking, such as PK-7K3Q9X-03."""
    if not 1 <= vehicle_number <= 99:
        raise ValueError("vehicle_number must be between 1 and 99.")
    return f"{booking_id}-{vehicle_number:02d}"
