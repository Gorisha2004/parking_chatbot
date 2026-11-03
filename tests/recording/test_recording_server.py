import asyncio
import json

import pytest
from starlette.testclient import TestClient

from parking_chatbot import config
from parking_chatbot.recording import server

VALID_TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {VALID_TOKEN}"}


def _call_tool(app, name: str, arguments: dict) -> dict:
    """Invoke a registered MCP tool directly, in-process - no HTTP, no session
    handshake, just the tool function itself (as registered) and its real
    validation/error handling. Tools are async at the SDK level; pytest here is
    sync, so each call gets its own short-lived event loop via asyncio.run()."""
    result = asyncio.run(app.mcp.call_tool(name, arguments))
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


VEHICLE = {"car_number": "AB123CD", "pass_id": "PK-7K3Q9X-01", "vehicle_type": "car"}
BOOKING = {
    "booking_id": "PK-7K3Q9X",
    "name": "Anna Smith",
    "period_start": "2026-11-20T09:00:00Z",
    "period_end": "2026-11-20T12:00:00Z",
    "approval_time": "2026-11-19T14:32:05Z",
    "vehicles": [VEHICLE],
}


@pytest.fixture(autouse=True)
def mcp_token(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", VALID_TOKEN)


@pytest.fixture
def client():
    # A fresh build_app() per test, not the shared module-level server.app: the
    # SDK's session manager can only be started once per instance, and each test
    # starts (and stops) one via TestClient's own lifespan handling.
    #
    # base_url="http://127.0.0.1" (not TestClient's default fake "testserver") so a
    # correctly authenticated request reaches the MCP protocol layer instead of
    # being rejected first by the SDK's own DNS-rebinding Host-header check - that
    # check is a separate concern from the auth tested here.
    with TestClient(server.build_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


# --- the bearer-auth middleware -------------------------------------------------------------------


def test_health_check_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_mcp_endpoint_rejects_a_request_with_no_token(client):
    assert client.post("/mcp", json={}).status_code == 401


def test_mcp_endpoint_rejects_a_wrong_token(client):
    response = client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401


def test_mcp_endpoint_rejects_a_malformed_authorization_header(client):
    response = client.post("/mcp", json={}, headers={"Authorization": VALID_TOKEN})  # missing "Bearer "
    assert response.status_code == 401


def test_mcp_endpoint_rejects_everything_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", None)
    assert client.post("/mcp", json={}, headers=AUTH).status_code == 401


def test_mcp_endpoint_accepts_the_correct_token(client):
    # Not a real MCP request body, so this won't succeed at the protocol level -
    # the point here is only that it gets past auth (never a 401).
    response = client.post("/mcp", json={}, headers=AUTH)
    assert response.status_code != 401


def test_rejection_response_never_includes_any_token(client):
    response = client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong-token-value"})
    assert "wrong-token-value" not in response.text
    assert VALID_TOKEN not in response.text


# --- the body-size-limit middleware ----------------------------------------------------------------


def test_mcp_endpoint_rejects_a_body_over_64kb(client):
    oversized = b"x" * (server.MAX_BODY_BYTES + 1)
    response = client.post("/mcp", content=oversized, headers=AUTH)
    assert response.status_code == 413


def test_mcp_endpoint_accepts_a_body_within_the_limit(client):
    response = client.post("/mcp", json={}, headers=AUTH)
    assert response.status_code != 413


# --- _require_configured_token: the startup refusal ----------------------------------------------


def test_require_configured_token_raises_when_unset(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", None)
    with pytest.raises(server.ServerMisconfiguredError):
        server._require_configured_token()


def test_require_configured_token_raises_when_empty(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", "")
    with pytest.raises(server.ServerMisconfiguredError):
        server._require_configured_token()


def test_require_configured_token_raises_when_too_short(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", "t" * 31)
    with pytest.raises(server.ServerMisconfiguredError):
        server._require_configured_token()


def test_require_configured_token_accepts_the_minimum_length(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", "t" * 32)
    server._require_configured_token()  # must not raise


def test_misconfigured_token_error_never_includes_the_token_value(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", "super-secret-but-too-short")
    with pytest.raises(server.ServerMisconfiguredError) as error:
        server._require_configured_token()
    assert "super-secret-but-too-short" not in str(error.value)


# --- main(): refuses to start, or starts on the configured host/port -----------------------------


def test_main_refuses_to_start_without_a_configured_token(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", None)
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: pytest.fail("must not start the server"))

    with pytest.raises(server.ServerMisconfiguredError):
        server.main()


def test_main_starts_uvicorn_on_the_host_and_port_from_mcp_server_url(monkeypatch):
    monkeypatch.setattr(config, "MCP_API_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(config, "MCP_SERVER_URL", "http://127.0.0.1:8002")
    calls = []
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))

    server.main()

    [kwargs] = calls
    assert kwargs == {"host": "127.0.0.1", "port": 8002}


# --- the record_reservation tool -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reservations_file(tmp_path, monkeypatch):
    """Point MCP_RESERVATIONS_FILE_PATH at a fresh temp file for every test in
    this module - the tool genuinely writes to disk, so this must never be the
    real configured path."""
    path = tmp_path / "reservations.txt"
    monkeypatch.setattr(config, "MCP_RESERVATIONS_FILE_PATH", path)
    return path


def test_record_reservation_tool_writes_the_expected_line(reservations_file):
    app = server.build_app()

    result = _call_tool(app, "record_reservation", BOOKING)

    assert result == {"booking_id": "PK-7K3Q9X", "recorded": True, "lines_written": 1}
    assert "Anna Smith | AB123CD" in reservations_file.read_text(encoding="utf-8")


def test_record_reservation_tool_is_a_no_op_on_a_repeated_booking_id(reservations_file):
    app = server.build_app()
    _call_tool(app, "record_reservation", BOOKING)
    first_contents = reservations_file.read_text(encoding="utf-8")

    result = _call_tool(app, "record_reservation", BOOKING)

    assert result == {"booking_id": "PK-7K3Q9X", "recorded": True, "lines_written": 0}
    assert reservations_file.read_text(encoding="utf-8") == first_contents


def test_record_reservation_tool_reports_invalid_data_without_writing_anything(reservations_file):
    app = server.build_app()
    bad_booking = {**BOOKING, "vehicles": [{**VEHICLE, "vehicle_type": "truck"}]}

    result = _call_tool(app, "record_reservation", bad_booking)

    assert result["recorded"] is False
    assert "error" in result
    assert not reservations_file.exists()


def test_record_reservation_tool_rejects_a_pipe_in_the_name_without_writing_anything(reservations_file):
    app = server.build_app()
    bad_booking = {**BOOKING, "name": "Anna|Smith"}

    result = _call_tool(app, "record_reservation", bad_booking)

    assert result["recorded"] is False
    assert "Anna|Smith" not in str(result)  # the rejected value itself is never echoed back
    assert not reservations_file.exists()


def test_record_reservation_tool_writes_a_whole_group_or_none_of_it(reservations_file):
    app = server.build_app()
    group = {
        **BOOKING,
        "vehicles": [
            VEHICLE,
            {"car_number": "ZZZ999", "pass_id": "PK-7K3Q9X-02", "vehicle_type": "truck"},  # invalid
        ],
    }

    result = _call_tool(app, "record_reservation", group)

    assert result["recorded"] is False
    assert not reservations_file.exists()
