"""Fixtures for the full-workflow integration tests.

The only two things faked anywhere in this package are exactly the two
stage4.md names as safe to fake: the LLM (RealToolChat below drives the real
Stage 1 tools directly, as if a model had chosen them) and the network hop to
the recording server (routed in-process into the real MCP tool). Everything
else - reservation validation and capacity checks, the SQL database, the
administrator's decision path, the MCP server's tool logic, and the reservation
file writer - is the real Stage 1-3 code.
"""
import json
from contextlib import closing
from dataclasses import dataclass, field

import pytest

from parking_chatbot import config
from parking_chatbot.chatbot import reservation_tools as rt
from parking_chatbot.db.database import get_connection
from parking_chatbot.orchestration.checkpointing import build_checkpointer
from parking_chatbot.orchestration.graph import build_workflow
from parking_chatbot.orchestration.nodes import WorkflowDependencies
from parking_chatbot.recording import client as recording_client
from parking_chatbot.recording import server as recording_server


@pytest.fixture
def capacity(sql_db):
    """Enough availability for every integration scenario's bookings."""
    with closing(get_connection()) as conn, conn:
        conn.execute("DELETE FROM availability")
        conn.execute("INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, 100, 100)")


@pytest.fixture
def reservation_file(tmp_path, monkeypatch):
    """A real, empty reservation file for this test only."""
    path = tmp_path / "reservations.txt"
    monkeypatch.setattr(config, "MCP_RESERVATIONS_FILE_PATH", path)
    return path


@pytest.fixture
def real_recording_server(monkeypatch, reservation_file):
    """Route the recording client straight into the real MCP tool, in-process.

    This is the one network hop stage4.md names as safe to fake: everything the
    real tool does once called - formatting, the dedup check, the file lock and
    append, the restrictive permissions - is the real Stage 3 code, writing to a
    real file. See tests/recording/test_recording_server.py for the same
    in-process call_tool() technique, used there for the same reason.
    """
    app = recording_server.build_app()

    async def _call_record_reservation(booking, timeout):
        result = await app.mcp.call_tool("record_reservation", recording_client._payload(booking))
        if result.is_error:
            raise RuntimeError(f"record_reservation reported an error for booking {booking.booking_id}")
        body = json.loads(result.content[0].text)
        if not body.get("recorded"):
            raise RuntimeError(f"record_reservation did not confirm booking {booking.booking_id}")
        return body

    monkeypatch.setattr(recording_client, "_call_record_reservation", _call_record_reservation)
    return app


@pytest.fixture
def broken_recording_server(monkeypatch, reservation_file):
    """Simulate the recording server being unreachable: every call fails, and
    nothing is ever written to the file."""

    async def _always_fails(booking, timeout):
        raise ConnectionError("the recording server is unreachable")

    monkeypatch.setattr(recording_client, "_call_record_reservation", _always_fails)


@dataclass
class RealToolChat:
    """A ChatFn for integration tests: each turn runs one real Stage 1
    reservation tool directly - as if an LLM had chosen it and its arguments -
    and returns that tool's own JSON response as the answer. Built with the
    sequence of (tool, kwargs) pairs this conversation's turns should run, one
    pair consumed per turn.

    This is the fake the LLM is allowed to be (stage4.md): the *decision* of
    which tool to call is scripted, but every call that follows runs the real
    Stage 1 validation, capacity check and database write.
    """

    steps: list[tuple]
    calls: list[str] = field(default_factory=list)

    def __call__(self, agent, user_message: str, thread_id: str) -> tuple[str, list[str]]:
        tool, kwargs = self.steps.pop(0)
        run_config = {"configurable": {"thread_id": thread_id}}
        answer = tool.invoke(dict(kwargs), run_config)
        self.calls.append(tool.name)
        return answer, [tool.name]


@pytest.fixture
def thread_id():
    import uuid

    tid = str(uuid.uuid4())
    yield tid
    rt._clear(tid)


@pytest.fixture
def real_tool_chat():
    """Factory for RealToolChat, so test modules need not import across test
    files (pytest's rootless test layout here has no package __init__.py to
    import through)."""
    return RealToolChat


@pytest.fixture
def workflow():
    """A fresh graph per test, with a real agent object placeholder (RealToolChat
    never uses it) and the real default accessors for everything else."""

    def _build(chat) -> object:
        deps = WorkflowDependencies(agent=object(), chat=chat)
        return build_workflow(deps=deps, checkpointer=build_checkpointer(backend="memory"))

    return _build
