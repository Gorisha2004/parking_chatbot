from datetime import datetime

import pytest
from pydantic import ValidationError

from parking_chatbot.admin.models import AdminDecision, ApprovalRequest, RequestVehicle


def _request(**overrides) -> ApprovalRequest:
    defaults = dict(
        booking_id="PK-AAAAAA",
        first_name="Anna",
        last_name="Smith",
        start=datetime(2026, 11, 20, 9, 0),
        end=datetime(2026, 11, 20, 12, 0),
        vehicles=[RequestVehicle(pass_id="PK-AAAAAA-01", car_number="AB123CD", vehicle_type="car")],
    )
    return ApprovalRequest(**{**defaults, **overrides})


def test_approval_request_holds_one_or_more_vehicles():
    request = _request(
        vehicles=[
            RequestVehicle(pass_id="PK-AAAAAA-01", car_number="AB123CD", vehicle_type="car"),
            RequestVehicle(pass_id="PK-AAAAAA-02", car_number="XY987ZT", vehicle_type="van"),
        ]
    )
    assert [v.pass_id for v in request.vehicles] == ["PK-AAAAAA-01", "PK-AAAAAA-02"]


def test_approval_request_requires_every_field():
    with pytest.raises(ValidationError):
        ApprovalRequest(booking_id="PK-AAAAAA")


def test_request_vehicle_allows_a_missing_vehicle_type_for_legacy_bookings():
    vehicle = RequestVehicle(pass_id="PK-AAAAAA-01", car_number="AB123CD", vehicle_type=None)
    assert vehicle.vehicle_type is None


def test_admin_decision_only_accepts_approved_or_refused():
    AdminDecision(booking_id="PK-AAAAAA", decision="approved")
    AdminDecision(booking_id="PK-AAAAAA", decision="refused")
    with pytest.raises(ValidationError):
        AdminDecision(booking_id="PK-AAAAAA", decision="maybe")


def test_admin_decision_reason_is_optional():
    decision = AdminDecision(booking_id="PK-AAAAAA", decision="refused")
    assert decision.reason is None
    decision = AdminDecision(booking_id="PK-AAAAAA", decision="refused", reason="no space that day")
    assert decision.reason == "no space that day"
