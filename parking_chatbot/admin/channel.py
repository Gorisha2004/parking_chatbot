"""How a completed reservation reaches the administrator.

Behind a Protocol so it can be swapped (and mocked in tests) without touching the
chat agent that calls it.
"""
from functools import lru_cache
from typing import Protocol

from parking_chatbot.admin.models import ApprovalRequest
from parking_chatbot.db import admin as admin_db


class AdminChannel(Protocol):
    def send_request(self, request: ApprovalRequest) -> None: ...


class DbAdminChannel:
    """The real channel: marks the booking escalated in the database.

    This is a local write, not a network call to the admin server - the chatbot
    must be able to submit a reservation whether or not that server happens to be
    running. The admin server reads escalated bookings whenever it is next polled.
    """

    def send_request(self, request: ApprovalRequest) -> None:
        admin_db.mark_escalated(request.booking_id)


class FakeAdminChannel:
    """Records every request in memory. No database, no network - for tests."""

    def __init__(self) -> None:
        self.sent: list[ApprovalRequest] = []

    def send_request(self, request: ApprovalRequest) -> None:
        self.sent.append(request)


@lru_cache(maxsize=1)
def get_admin_channel() -> AdminChannel:
    """The channel the live chatbot uses. Tests monkeypatch this to inject a fake,
    the same pattern used for get_vector_store() and get_chat_model()."""
    return DbAdminChannel()
