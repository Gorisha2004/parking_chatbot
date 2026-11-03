"""The three Stage 4 load-test scenarios, one per component named in stage4.md:
the chatbot in interactive dialogue mode, administrator confirmation, and MCP
server recording and storage.

Each scenario is a thin driver around the real component - the same functions
the orchestration nodes call in production - wired through load.run_concurrent.
No scenario here talks to a real LLM unless explicitly asked to (`use_live_llm`
on the chatbot scenario): a fake agent is used by default, so the numbers are
reproducible without network access or API cost, and the report says plainly
which mode produced them.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import patch

from parking_chatbot import config
from parking_chatbot.admin.agent import apply_decision
from parking_chatbot.chatbot.agent import chat as chatbot_chat
from parking_chatbot.db.reservations import create_reservation
from parking_chatbot.evaluation.load import LoadResult, run_concurrent
from parking_chatbot.reservation.ids import new_booking_id, pass_id
from parking_chatbot.reservation.models import ReservationDraft, Vehicle
from parking_chatbot.recording import server as recording_server

DEFAULT_DIALOGUE_TURNS = (
    "What are your opening hours?",
    "Do you have space for a van?",
    "What is the daily rate?",
)


@dataclass(frozen=True)
class _FakeToolMessage:
    type: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class _FakeChatModelAgent:
    """Stands in for the real LangGraph agent in build_agent(): same `.invoke()`
    shape chatbot.agent._invoke_agent expects, no model call. Exercises exactly
    chatbot.agent.chat()'s own overhead (guardrails, message bookkeeping) under
    concurrent, independent conversations - not the LLM's latency, which this
    scenario explicitly does not claim to measure in its default mode."""

    reply: str = "Our office is open 7am to 9pm, seven days a week."

    def invoke(self, payload: dict, config: dict) -> dict:
        human = payload["messages"][0]["content"]
        return {"messages": [_FakeToolMessage("human", human), _FakeToolMessage("ai", self.reply)]}


def chatbot_dialogue_scenario(
    sessions: int,
    concurrency: int,
    turns_per_session: int = len(DEFAULT_DIALOGUE_TURNS),
    use_live_llm: bool = False,
) -> LoadResult:
    """Load-test the chatbot in interactive dialogue mode: `sessions` independent
    conversations, each sending `turns_per_session` messages in order, up to
    `concurrency` conversations running at once - the same shape as several
    users talking to the CLI at the same time.

    Each call measured is one whole session (all of its turns), not one turn:
    that is what "interactive dialogue mode" means here, as opposed to a single
    one-shot question.
    """
    if use_live_llm:
        from parking_chatbot.chatbot.agent import build_agent

        agent = build_agent()
    else:
        agent = _FakeChatModelAgent()

    turns = DEFAULT_DIALOGUE_TURNS[:turns_per_session] if turns_per_session <= len(DEFAULT_DIALOGUE_TURNS) else (
        DEFAULT_DIALOGUE_TURNS * (turns_per_session // len(DEFAULT_DIALOGUE_TURNS) + 1)
    )[:turns_per_session]

    def _one_session(index: int) -> None:
        thread_id = f"load-chat-{index}-{uuid.uuid4()}"
        for turn in turns:
            chatbot_chat(agent, turn, thread_id)

    return run_concurrent(_one_session, total_calls=sessions, concurrency=concurrency)


def _insert_pending_booking(index: int) -> str:
    """A real pending booking, saved exactly as Stage 1's submit_reservation
    saves one - so the admin-confirmation scenario below decides on real rows,
    not a hand-built fixture shape that could drift from the real schema."""
    start = datetime.now() + timedelta(days=1)
    draft = ReservationDraft(
        first_name="Load",
        last_name="Test",
        start=start,
        end=start + timedelta(hours=2),
        vehicles=[Vehicle(car_number=f"LD{index % 10000:04d}AA", vehicle_type="car")],
    )
    return create_reservation(draft).booking_id


def admin_confirmation_scenario(decisions: int, concurrency: int, approve_fraction: float = 0.7) -> LoadResult:
    """Load-test administrator confirmation: `decisions` pending bookings are
    created, then each is decided (approved or refused) concurrently through
    apply_decision - the one real write path Stage 2 ever uses, called the same
    way the admin agent's record_decision tool and the REST API both call it.

    No LLM is involved in this path at all (apply_decision is deterministic), so
    there is no fake-vs-live distinction to make here; this measures the real
    database write path, including SQLite lock contention at this concurrency.

    An approval also fires apply_decision's own Stage 3 recording hook - real
    network I/O to whatever MCP_SERVER_URL happens to be, which this scenario is
    not trying to measure (that is mcp_recording_scenario's job) and which would
    otherwise make these numbers depend on an unrelated server being up. It is
    patched out to a no-op for the duration of this scenario only.
    """
    booking_ids = [_insert_pending_booking(i) for i in range(decisions)]

    def _decide(index: int) -> None:
        booking_id = booking_ids[index]
        decision = "approved" if index % 10 < approve_fraction * 10 else "refused"
        reason = None if decision == "approved" else "load test refusal"
        apply_decision(booking_id, decision, reason)

    with patch("parking_chatbot.recording.client.send_approved_booking"):
        return run_concurrent(_decide, total_calls=decisions, concurrency=concurrency)


def mcp_recording_scenario(bookings: int, concurrency: int) -> tuple[LoadResult, dict]:
    """Load-test MCP server recording and storage: `bookings` approved bookings
    are recorded concurrently through the real MCP tool (recording.server's
    record_reservation, called in-process the same way the recording client's
    _call_record_reservation calls it over the network - see
    tests/recording/test_recording_server.py for the same technique), writing
    to a real reservation file guarded by its real file lock.

    Returns (load_result, integrity) - integrity reports whether the file ended
    up with exactly one line per vehicle and no corruption, which a pure latency
    number cannot show on its own.
    """
    app = recording_server.build_app()
    path = config.MCP_RESERVATIONS_FILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    def _record(index: int) -> None:
        # A real, validly-formatted booking/pass ID (new_booking_id()'s alphabet
        # excludes look-alike characters) - not an arbitrary string, so this
        # exercises the real validation path rather than always failing it.
        booking_id = new_booking_id()
        payload = {
            "booking_id": booking_id,
            "name": "Load Test",
            "period_start": "2026-11-20T09:00:00Z",
            "period_end": "2026-11-20T12:00:00Z",
            "approval_time": "2026-11-19T14:32:00Z",
            "vehicles": [{"car_number": f"LD{index % 10000:04d}AA", "pass_id": pass_id(booking_id, 1), "vehicle_type": "car"}],
        }
        result = asyncio.run(app.mcp.call_tool("record_reservation", payload))
        if result.is_error:
            raise RuntimeError(f"record_reservation reported an error for {booking_id}")
        body = json.loads(result.content[0].text)
        if not body.get("recorded"):
            raise RuntimeError(f"record_reservation did not confirm {booking_id}")

    result = run_concurrent(_record, total_calls=bookings, concurrency=concurrency)

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    successes = len(result.outcomes) - len(result.failures)
    integrity = {
        "lines_written": len(lines),
        "expected_lines": successes,  # one vehicle per successful booking
        "file_ends_with_newline": path.exists() and path.read_bytes().endswith(b"\n"),
        "matches": len(lines) == successes,
    }
    return result, integrity
