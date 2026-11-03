"""The MCP server (Stage 3): a streamable-HTTP server that appends approved
reservations to a text file.

Deliberately not using the mcp SDK's own auth support (TokenVerifier/AuthSettings):
that machinery is built for a real OAuth authorization server issuing short-lived
tokens to many clients, with its own discovery metadata endpoints. What STAGE_3.md
actually asks for is much simpler - one shared, long-lived secret between our own
client and server - so auth here is a small, explicit piece of HTTP middleware
instead, the same shape as admin/server.py's bearer check.
"""
import logging
from secrets import compare_digest
from urllib.parse import urlparse

import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from parking_chatbot import config
from parking_chatbot.recording import writer
from parking_chatbot.recording.formatting import LineFormatError, VehicleLineInput

logger = logging.getLogger(__name__)

# A short, guessable token would defeat the whole point of requiring one.
MIN_TOKEN_LENGTH = 32
_PUBLIC_PATHS = frozenset({"/health"})

# A reservation group is a handful of short text fields for at most
# MAX_VEHICLES_PER_BOOKING vehicles - nowhere near this size. Anything bigger is a
# mistake or an attempt to exhaust memory, not a legitimate request.
MAX_BODY_BYTES = 64 * 1024


class LimitBodySize(BaseHTTPMiddleware):
    """Rejects any request whose declared Content-Length exceeds MAX_BODY_BYTES,
    before the body is ever read into memory."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)


class ServerMisconfiguredError(RuntimeError):
    """The recording server must not start without a properly configured bearer
    token - checked before uvicorn ever binds a port, not just when the first
    request arrives."""


def _require_configured_token() -> None:
    token = config.MCP_API_TOKEN
    if not token or len(token) < MIN_TOKEN_LENGTH:
        raise ServerMisconfiguredError(
            f"MCP_API_TOKEN must be set to a string of at least {MIN_TOKEN_LENGTH} characters."
        )


def _bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header or not authorization_header.startswith("Bearer "):
        return None
    return authorization_header.removeprefix("Bearer ")


class RequireBearerToken(BaseHTTPMiddleware):
    """Rejects every request except /health unless it carries the exact configured
    bearer token, checked in constant time. Fails closed: if no token is configured,
    every request is rejected. Never logs or echoes a token, provided or expected,
    anywhere - including in this rejection response."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        expected = config.MCP_API_TOKEN
        provided = _bearer_token(request.headers.get("authorization"))
        if not expected or not provided or not compare_digest(provided, expected):
            logger.warning("Rejected unauthenticated request to %s", request.url.path)
            return JSONResponse({"detail": "Missing or invalid token"}, status_code=401)
        return await call_next(request)


class VehicleInput(BaseModel):
    """One vehicle's data, as the tool receives it over MCP (JSON) - converted to
    formatting.py's VehicleLineInput before any validation or writing happens."""

    car_number: str
    pass_id: str
    vehicle_type: str


def build_app():
    """Build a fresh MCP server and its ASGI app.

    A factory, not a bare module-level app: the SDK's streamable-HTTP session
    manager can only be started (via the app's lifespan) once per instance - the
    real server only ever needs to start once anyway, but each test that spins up
    its own TestClient needs its own fresh instance.
    """
    mcp = MCPServer("CityPark Central Recording Server")

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @mcp.tool()
    def record_reservation(
        booking_id: str,
        name: str,
        period_start: str,
        period_end: str,
        approval_time: str,
        vehicles: list[VehicleInput],
    ) -> dict:
        """Append one line per vehicle for an approved reservation to the
        reservation file - or do nothing if this booking ID is already recorded.

        Only ever call this for a reservation an administrator has actually
        approved - this tool has no way to check a booking's real status itself,
        so that decision must already have been made by the caller."""
        try:
            result = writer.record_group(
                booking_id=booking_id,
                name=name,
                period_start=period_start,
                period_end=period_end,
                approval_time=approval_time,
                vehicles=[
                    VehicleLineInput(car_number=v.car_number, pass_id=v.pass_id, vehicle_type=v.vehicle_type)
                    for v in vehicles
                ],
            )
        except LineFormatError as error:
            logger.warning("Rejected invalid reservation data for booking %s (field: %s)", booking_id, error.field)
            return {"booking_id": booking_id, "recorded": False, "error": str(error)}
        return {"booking_id": booking_id, **result}

    app = mcp.streamable_http_app()
    app.add_middleware(RequireBearerToken)
    app.add_middleware(LimitBodySize)
    app.mcp = mcp  # lets tests call tools directly (app.mcp.call_tool(...)),
    # without driving a full MCP session handshake just to test business logic.
    return app


app = build_app()


def main() -> None:
    """Entry point for scripts/run_mcp_server.py."""
    _require_configured_token()
    parsed = urlparse(config.MCP_SERVER_URL)
    uvicorn.run(app, host=parsed.hostname, port=parsed.port)
