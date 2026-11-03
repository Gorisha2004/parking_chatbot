"""Fakes shared by the orchestration tests.

Nothing here touches an LLM, the admin channel or the recording server: every
dependency the nodes use is injected, so the whole workflow can be driven from
plain Python objects.
"""
from dataclasses import dataclass, field
from typing import Any

import pytest

from parking_chatbot.orchestration.nodes import WorkflowDependencies
from parking_chatbot.recording.client import RecordingError

BOOKING_ID = "PK-AAAAAA"
VEHICLES = [
    {"pass_id": f"{BOOKING_ID}-01", "car_number": "AB12CD", "vehicle_type": "car"},
    {"pass_id": f"{BOOKING_ID}-02", "car_number": "EF34GH", "vehicle_type": "van"},
]


@dataclass
class FakeDraft:
    """Stand-in for a ReservationDraft: only model_dump() is ever called on it."""

    data: dict = field(default_factory=lambda: {"first_name": "Anna"})

    def model_dump(self, mode: str = "python") -> dict:
        return dict(self.data)


@dataclass
class FakeChat:
    """Stand-in for chatbot.agent.chat: a scripted answer and tool list per call."""

    answer: str = "Here is what I found."
    tools: list[str] = field(default_factory=list)
    calls: list[tuple[Any, str, str]] = field(default_factory=list)

    def __call__(self, agent: Any, message: str, thread_id: str) -> tuple[str, list[str]]:
        self.calls.append((agent, message, thread_id))
        return self.answer, list(self.tools)


@dataclass
class FakeBookingStore:
    """Stand-in for db.admin.get_booking, plus a scriptable decision."""

    status: str = "pending"
    decision_reason: str | None = None
    vehicles: list[dict] = field(default_factory=lambda: [dict(v) for v in VEHICLES])
    known: bool = True
    reads: list[str] = field(default_factory=list)

    def __call__(self, booking_id: str) -> dict | None:
        self.reads.append(booking_id)
        if not self.known:
            return None
        return {
            "booking_id": booking_id,
            "status": self.status,
            "decided_at": None,
            "decision_reason": self.decision_reason,
            "vehicles": [dict(v) for v in self.vehicles],
        }


@dataclass
class FakeRecorder:
    """Stand-in for recording.client.record_approved_booking."""

    fail: bool = False
    already_recorded: bool = False
    calls: list[str] = field(default_factory=list)

    def __call__(self, booking_id: str) -> bool:
        self.calls.append(booking_id)
        if self.fail:
            raise RecordingError(f"Could not record booking {booking_id} (fake failure)")
        return not self.already_recorded


@dataclass
class Fakes:
    """One bundle so a test can both build dependencies and inspect the fakes.

    Also carries the sample booking ID and a draft factory, so a test module never
    has to import from this conftest (which is not an importable package).
    """

    chat: FakeChat
    bookings: FakeBookingStore
    recorder: FakeRecorder
    draft: FakeDraft | None = None
    last_booking_id: str | None = BOOKING_ID
    booking_id: str = BOOKING_ID
    vehicles: list[dict] = field(default_factory=lambda: [dict(v) for v in VEHICLES])

    @staticmethod
    def make_draft(data: dict | None = None) -> FakeDraft:
        return FakeDraft(data if data is not None else {"first_name": "Anna"})

    def deps(self) -> WorkflowDependencies:
        return WorkflowDependencies(
            agent=object(),
            chat=self.chat,
            read_draft=lambda thread_id: self.draft,
            read_last_booking_id=lambda thread_id: self.last_booking_id,
            read_booking=self.bookings,
            record_booking=self.recorder,
        )


@pytest.fixture
def fakes() -> Fakes:
    return Fakes(chat=FakeChat(), bookings=FakeBookingStore(), recorder=FakeRecorder())
