from datetime import datetime

from parking_chatbot.admin import channel as channel_module
from parking_chatbot.admin.channel import DbAdminChannel, FakeAdminChannel, get_admin_channel
from parking_chatbot.admin.models import ApprovalRequest, RequestVehicle
from parking_chatbot.db import admin as admin_db


def _request(booking_id: str = "PK-AAAAAA") -> ApprovalRequest:
    return ApprovalRequest(
        booking_id=booking_id,
        first_name="Anna",
        last_name="Smith",
        start=datetime(2026, 11, 20, 9, 0),
        end=datetime(2026, 11, 20, 12, 0),
        vehicles=[RequestVehicle(pass_id=f"{booking_id}-01", car_number="AB123CD", vehicle_type="car")],
    )


def test_fake_channel_records_every_request_in_memory():
    fake = FakeAdminChannel()
    fake.send_request(_request("PK-AAAAAA"))
    fake.send_request(_request("PK-BBBBBB"))
    assert [r.booking_id for r in fake.sent] == ["PK-AAAAAA", "PK-BBBBBB"]


def test_db_channel_marks_the_booking_escalated(sql_db):
    with_conn_insert = """
        INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time)
        VALUES ('PK-AAAAAA', 'Anna', 'Smith', '2026-11-20T09:00', '2026-11-20T12:00')
    """
    from contextlib import closing

    from parking_chatbot.db.database import get_connection

    with closing(get_connection()) as conn, conn:
        conn.execute(with_conn_insert)

    DbAdminChannel().send_request(_request("PK-AAAAAA"))

    assert admin_db.list_pending_bookings()[0].booking_id == "PK-AAAAAA"


def test_get_admin_channel_is_cached_to_one_instance():
    channel_module.get_admin_channel.cache_clear()
    assert get_admin_channel() is get_admin_channel()
    channel_module.get_admin_channel.cache_clear()
