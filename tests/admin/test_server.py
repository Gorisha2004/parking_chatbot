from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from parking_chatbot import config
from parking_chatbot.admin import agent as agent_module
from parking_chatbot.admin.server import app
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.database import get_connection

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def admin_token(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", "test-token")


@pytest.fixture
def client(sql_db):
    # sql_db must be set up first (it points config.SQL_DB_PATH at a tmp file) so the
    # lifespan startup below - entered via the `with` - runs its migration against
    # that tmp file, never against the real database. Depending on it here, rather
    # than relying on each test's own parameter order, makes that ordering certain.
    with TestClient(app) as test_client:
        yield test_client


def _insert_pending_booking(booking_id: str = "PK-AAAAAA", created_at: str | None = None) -> None:
    with closing(get_connection()) as conn, conn:
        if created_at is None:
            cursor = conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, escalated_at) "
                "VALUES (?, 'Anna', 'Smith', '2026-11-20T09:00', '2026-11-20T12:00', '2026-11-18T00:00')",
                (booking_id,),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, "
                "escalated_at, created_at) VALUES (?, 'Anna', 'Smith', '2026-11-20T09:00', "
                "'2026-11-20T12:00', '2026-11-18T00:00', ?)",
                (booking_id, created_at),
            )
        conn.execute(
            "INSERT INTO reservation_vehicles (reservation_id, pass_id, car_number, vehicle_type) "
            "VALUES (?, ?, 'AB123CD', 'car')",
            (cursor.lastrowid, f"{booking_id}-01"),
        )


def test_health_check_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_openapi_schema_declares_bearer_auth_on_protected_routes_only(client):
    # This is what actually drives /docs: the Authorize button only appears, and
    # only attaches the token to requests, for paths listed with this security
    # requirement against this scheme.
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"] == {"HTTPBearer": {"type": "http", "scheme": "bearer"}}
    assert schema["paths"]["/requests"]["get"]["security"] == [{"HTTPBearer": []}]
    assert schema["paths"]["/requests/{booking_id}/decision"]["post"]["security"] == [{"HTTPBearer": []}]
    assert "security" not in schema["paths"]["/health"]["get"]


def test_a_token_of_a_different_length_is_rejected_not_crashed_on(client):
    # secrets.compare_digest must handle mismatched lengths safely; a naive
    # constant-time implementation can crash or misbehave on this.
    response = client.get("/requests", headers={"Authorization": "Bearer short"})
    assert response.status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "wrong-shape"}])
def test_requests_endpoint_rejects_missing_or_wrong_token(client, headers, sql_db):
    assert client.get("/requests", headers=headers).status_code == 401


def test_requests_endpoint_rejects_everything_when_no_token_is_configured(client, monkeypatch, sql_db):
    monkeypatch.setattr(config, "ADMIN_API_TOKEN", None)
    assert client.get("/requests", headers=AUTH).status_code == 401


def test_list_requests_returns_pending_bookings_with_a_summary(client, sql_db):
    _insert_pending_booking("PK-AAAAAA")

    response = client.get("/requests", headers=AUTH)

    assert response.status_code == 200
    [item] = response.json()
    assert item["request"]["booking_id"] == "PK-AAAAAA"
    assert "PK-AAAAAA" in item["summary"]


def test_approve_decision_happy_path(client, sql_db, monkeypatch):
    _insert_pending_booking("PK-AAAAAA")
    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "approve, looks fine"})

    assert response.status_code == 200
    assert response.json() == {"booking_id": "PK-AAAAAA", "decision": "approved", "reason": "looks fine"}
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"


def test_refuse_decision_happy_path(client, sql_db):
    _insert_pending_booking("PK-AAAAAA")
    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "refuse - no space"})

    assert response.status_code == 200
    assert response.json()["decision"] == "refused"
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "refused"


def test_decision_on_unknown_booking_is_404(client, sql_db):
    response = client.post("/requests/PK-NOPE00/decision", headers=AUTH, json={"reply": "approve"})
    assert response.status_code == 404


def test_decision_on_an_already_decided_booking_is_409(client, sql_db):
    _insert_pending_booking("PK-AAAAAA")
    client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "approve"})

    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "refuse"})

    assert response.status_code == 409
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "approved"  # first decision still stands


def test_decision_on_an_expired_booking_is_409_with_a_distinct_message(client, sql_db, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_REQUEST_EXPIRY_MINUTES", 60)
    overdue = (datetime.now(timezone.utc) - timedelta(minutes=61)).strftime("%Y-%m-%d %H:%M:%S")
    _insert_pending_booking("PK-AAAAAA", created_at=overdue)

    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "approve"})

    assert response.status_code == 409
    assert "expired" in response.json()["detail"]
    assert "already" not in response.json()["detail"]  # distinct wording from "already decided"


def test_a_repeated_identical_decision_is_idempotent_not_409(client, sql_db):
    _insert_pending_booking("PK-AAAAAA")
    client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "approve, fine"})

    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "approve, fine"})

    assert response.status_code == 200
    assert response.json()["decision"] == "approved"


def test_server_startup_migrates_an_old_schema_database_without_losing_data(sql_db):
    """Reproduces a real bug: GET /requests failed with 'no such column: escalated_at'
    because nothing ever called create_tables() against a database that predates the
    Stage 2 columns. Starting the server (the `with` below triggers its real startup
    lifespan) must migrate it automatically and keep existing rows."""
    pre_stage2_layout = (
        "CREATE TABLE reservations (id INTEGER PRIMARY KEY AUTOINCREMENT, booking_id TEXT NOT NULL UNIQUE, "
        "first_name TEXT NOT NULL, last_name TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    with closing(get_connection()) as conn, conn:
        conn.execute("DROP TABLE reservation_vehicles")
        conn.execute("DROP TABLE reservations")
        conn.execute(pre_stage2_layout)
        conn.execute(
            "INSERT INTO reservations (booking_id, first_name, last_name, start_time, end_time, status) "
            "VALUES ('PK-OLD001', 'Old', 'Booking', '2026-11-20T09:00', '2026-11-20T12:00', 'approved')"
        )

    with TestClient(app) as client:
        response = client.get("/requests", headers=AUTH)

    assert response.status_code == 200  # was a 500: "no such column: escalated_at"
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT first_name, status FROM reservations WHERE booking_id = 'PK-OLD001'"
        ).fetchone()
    assert row == ("Old", "approved")  # the pre-existing row is untouched


def test_ambiguous_reply_is_422_and_does_not_change_the_booking(client, sql_db, monkeypatch):
    _insert_pending_booking("PK-AAAAAA")
    monkeypatch.setattr(agent_module, "_interpret_unclear_reply", lambda reply: (_ for _ in ()).throw(
        agent_module.AmbiguousReplyError(reply)
    ))

    response = client.post("/requests/PK-AAAAAA/decision", headers=AUTH, json={"reply": "hmm not sure"})

    assert response.status_code == 422
    assert admin_db.get_booking("PK-AAAAAA")["status"] == "pending"
