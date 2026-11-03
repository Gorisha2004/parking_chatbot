"""Client for the Stage 3 recording server: a best-effort hook called right after
an approval, and a retry sweep for approvals the hook (or the server) missed.
"""
import asyncio
import json
import logging
import threading
from datetime import timezone
from zoneinfo import ZoneInfo

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from parking_chatbot import config
from parking_chatbot.admin.models import ApprovedBooking
from parking_chatbot.db import admin as admin_db

logger = logging.getLogger(__name__)

HOOK_TIMEOUT_SECONDS = 3


def _to_utc_iso(naive_local_dt) -> str:
    return naive_local_dt.replace(tzinfo=ZoneInfo(config.FACILITY_TIMEZONE)).astimezone(timezone.utc).isoformat()


def _payload(booking: ApprovedBooking) -> dict:
    return {
        "booking_id": booking.booking_id,
        "name": f"{booking.first_name} {booking.last_name}",
        "period_start": _to_utc_iso(booking.start),
        "period_end": _to_utc_iso(booking.end),
        "approval_time": _to_utc_iso(booking.decided_at),
        "vehicles": [
            {"car_number": v.car_number, "pass_id": v.pass_id, "vehicle_type": v.vehicle_type}
            for v in booking.vehicles
        ],
    }


def _mcp_endpoint_url() -> str:
    """MCP_SERVER_URL is documented as the bare server origin (no path) - but if
    it was set to the full endpoint instead (...:8002/mcp), appending "/mcp" again
    would silently send every request to a path that doesn't exist (.../mcp/mcp,
    a 404), with nothing to show for it but a generic connection-ish failure. This
    tolerates either form instead of relying on .env being exactly right."""
    base = config.MCP_SERVER_URL.rstrip("/")
    return base if base.endswith("/mcp") else f"{base}/mcp"


async def _call_record_reservation(booking: ApprovedBooking, timeout: float) -> dict:
    """Raises on any transport/protocol/tool-reported error - callers decide
    whether that's fatal (never, for the hook) or just "try again later" (the sweep)."""
    http_client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {config.MCP_API_TOKEN}"}, timeout=timeout)
    async with streamable_http_client(_mcp_endpoint_url(), http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("record_reservation", _payload(booking))
    if result.is_error:
        raise RuntimeError(f"record_reservation reported an error for booking {booking.booking_id}")
    body = json.loads(result.content[0].text)
    if not body.get("recorded"):
        raise RuntimeError(f"record_reservation did not confirm booking {booking.booking_id}")
    return body


class RecordingError(RuntimeError):
    """The recording server could not be reached, or did not confirm the write.

    Raised only by record_approved_booking(). The best-effort hook below swallows
    it; the Stage 4 orchestration layer catches it to route the booking down its
    "recording failed" path and tell the user something concrete.
    """


def record_approved_booking(booking_id: str, timeout: float = HOOK_TIMEOUT_SECONDS) -> bool:
    """Record one approved booking, and raise RecordingError if that fails.

    The strict counterpart to send_approved_booking() below: same single attempt,
    same write, but the failure is reported instead of logged and dropped, so a
    caller that needs to know (a graph node deciding which message the user gets)
    can tell the difference between "recorded" and "not recorded yet".

    Returns True if this call recorded the booking, and False if there was nothing
    left to record - an unknown booking, one that is not approved, or one already
    recorded. All three are already-consistent states, not failures.
    """
    bookings = [b for b in admin_db.list_unrecorded_approved_bookings() if b.booking_id == booking_id]
    if not bookings:
        return False
    try:
        asyncio.run(_call_record_reservation(bookings[0], timeout))
    except Exception as error:
        raise RecordingError(
            f"Could not record booking {booking_id} ({type(error).__name__}: {error})"
        ) from error
    admin_db.mark_recorded(booking_id)
    return True


def send_approved_booking(booking_id: str, timeout: float = HOOK_TIMEOUT_SECONDS) -> None:
    """Best-effort: called right after an approval is written. Never raises -
    failure here just leaves recorded_at empty, and retry_unrecorded() (run on the
    next admin action and on an interval) picks it up later. No backoff here by
    design: this is a single, short, one-shot attempt, not a retry loop."""
    try:
        record_approved_booking(booking_id, timeout)
    except Exception:
        logger.warning("Could not record booking %s with the recording server yet", booking_id)


def retry_unrecorded() -> None:
    """Sweep every approved-but-unrecorded booking and retry it. Backoff lives
    here, not in send_approved_booking: a booking that keeps failing is simply
    retried again next sweep (MCP_RETRY_INTERVAL_SECONDS apart) rather than
    hammered in a tight loop within one sweep."""
    for booking in admin_db.list_unrecorded_approved_bookings():
        try:
            asyncio.run(_call_record_reservation(booking, config.MCP_REQUEST_TIMEOUT_SECONDS))
        except Exception as error:
            logger.warning(
                "Retry failed for booking %s; will try again next sweep (%s: %s)",
                booking.booking_id,
                type(error).__name__,
                error,
            )
            continue
        admin_db.mark_recorded(booking.booking_id)


def start_retry_sweep_thread() -> threading.Event:
    """Run retry_unrecorded() once immediately, then again every
    MCP_RETRY_INTERVAL_SECONDS on a daemon thread. Used by admin_cli.py and
    run_admin_server.py so retries happen on startup and on an interval while
    either is running. Returns the stop_event; set it to stop the thread."""
    stop_event = threading.Event()

    def _sweep_once() -> None:
        try:
            retry_unrecorded()
        except Exception as error:
            logger.warning("Retry sweep failed; will try again next interval (%s: %s)", type(error).__name__, error)

    def _loop() -> None:
        _sweep_once()
        while not stop_event.wait(config.MCP_RETRY_INTERVAL_SECONDS):
            _sweep_once()

    threading.Thread(target=_loop, daemon=True).start()
    return stop_event
