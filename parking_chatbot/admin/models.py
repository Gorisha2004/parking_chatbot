"""The message contract between the chat agent and the admin-approval agent.

Deliberately separate from reservation.models: this is what crosses the boundary
between the two agents, not the full internal reservation/draft representation.
Carries only what an administrator needs to decide - never other users' data,
and nothing from elsewhere in the conversation.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class RequestVehicle(BaseModel):
    """One vehicle's pass, as shown to the administrator."""

    pass_id: str
    car_number: str
    vehicle_type: str | None  # None only for vehicles saved before vehicle types existed


class ApprovalRequest(BaseModel):
    """A booking handed to the administrator for one decision (the whole group, not per vehicle)."""

    booking_id: str
    first_name: str
    last_name: str
    start: datetime
    end: datetime
    vehicles: list[RequestVehicle]


class AdminDecision(BaseModel):
    """The administrator's decision on one booking."""

    booking_id: str
    decision: Literal["approved", "refused"]
    reason: str | None = None


class ApprovedBooking(BaseModel):
    """An approved booking not yet confirmed written to the reservation file
    (Stage 3) - everything recording needs: who, when approved, and every vehicle."""

    booking_id: str
    first_name: str
    last_name: str
    start: datetime
    end: datetime
    decided_at: datetime
    vehicles: list[RequestVehicle]
