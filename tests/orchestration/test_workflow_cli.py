"""Tests for the single entry point that runs the whole system.

The graph is built from the same injected fakes the other orchestration tests
use, and stdin is scripted, so a whole session runs without an LLM, an
administrator, a recording server or a terminal.
"""
import threading

import pytest

from parking_chatbot.orchestration import cli
from parking_chatbot.orchestration.checkpointing import build_checkpointer
from parking_chatbot.orchestration.graph import build_workflow, run_turn, workflow_state


@pytest.fixture
def session(fakes):
    workflow = build_workflow(deps=fakes.deps(), checkpointer=build_checkpointer(backend="memory"))
    return cli.WorkflowSession(workflow, "conv-1")


def _script(monkeypatch, *lines):
    """Feed the input loop a fixed list of user inputs."""
    remaining = list(lines)

    def fake_input(prompt=""):
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


# --- argument parsing ---------------------------------------------------------


def test_parse_args_defaults_to_no_debug_and_no_fixed_conversation():
    args = cli.parse_args([])
    assert args.debug is False
    assert args.thread_id is None
    assert args.checkpoint_backend is None


def test_parse_args_accepts_a_conversation_to_resume_and_a_backend_override():
    args = cli.parse_args(["--debug", "--thread-id", "conv-7", "--checkpoint-backend", "memory"])
    assert (args.debug, args.thread_id, args.checkpoint_backend) == (True, "conv-7", "memory")


def test_parse_args_rejects_an_unsupported_backend():
    with pytest.raises(SystemExit):
        cli.parse_args(["--checkpoint-backend", "redis"])


# --- rendering ----------------------------------------------------------------


def test_render_puts_the_answer_before_any_notice():
    lines = cli.render({"response": "We open at 6am.", "notices": ["Booking PK-AAAAAA was refused."]})
    assert lines == ["We open at 6am.", "Booking PK-AAAAAA was refused."]


def test_render_of_a_tick_with_nothing_to_say_is_empty():
    assert cli.render({"response": "", "notices": []}) == []
    assert cli.render({}) == []


# --- the session --------------------------------------------------------------


def test_send_runs_a_turn_and_returns_what_to_print(session, fakes):
    fakes.chat.answer = "We open at 6am."
    assert session.send("when do you open?") == ["We open at 6am."]


def test_tick_is_skipped_entirely_when_nothing_is_waiting(session, fakes):
    assert session.tick() == []
    assert fakes.bookings.reads == []  # no database read, and no LLM call
    assert fakes.chat.calls == []


def test_tick_reports_a_decision_that_landed_between_turns(session, fakes):
    fakes.chat.tools = ["submit_reservation"]
    session.send("yes, confirm")
    fakes.chat.tools = []

    fakes.bookings.status = "approved"
    lines = session.tick()
    assert any("approved and recorded" in line for line in lines)


def test_the_session_serializes_the_input_loop_and_the_ticker(session, fakes):
    """One checkpointed thread must only ever have one writer at a time."""
    fakes.chat.tools = ["submit_reservation"]
    session.send("yes, confirm")
    fakes.chat.tools = []
    fakes.bookings.status = "approved"

    results: list[list[str]] = []
    threads = [
        threading.Thread(target=lambda: results.append(session.tick())),
        threading.Thread(target=lambda: results.append(session.send("thanks"))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    # The decision is announced exactly once, whichever thread got there first.
    announcements = [line for lines in results for line in lines if "approved and recorded" in line]
    assert len(announcements) == 1


# --- the input loop -----------------------------------------------------------


def test_run_loop_exits_on_a_quit_command(session, monkeypatch, capsys):
    _script(monkeypatch, "hello", "exit", "never read")
    cli.run_loop(session)
    assert "Bye!" in capsys.readouterr().out


def test_run_loop_exits_cleanly_on_end_of_input(session, monkeypatch, capsys):
    _script(monkeypatch, "hello")
    cli.run_loop(session)
    assert "Bye!" in capsys.readouterr().out


def test_run_loop_ignores_empty_input(session, fakes, monkeypatch):
    _script(monkeypatch, "", "   ", "exit")
    cli.run_loop(session)
    assert fakes.chat.calls == []


def test_run_loop_prints_the_answer_and_the_notice(session, fakes, monkeypatch, capsys):
    fakes.chat.answer = "Noted."
    fakes.chat.tools = ["submit_reservation"]
    _script(monkeypatch, "yes, confirm", "exit")
    cli.run_loop(session)

    out = capsys.readouterr().out
    assert "Bot: Noted." in out
    assert "sent to an administrator" in out


def test_run_loop_reports_a_failed_turn_without_losing_the_session(session, fakes, monkeypatch, capsys):
    def explode(agent, message, thread_id):
        raise RuntimeError("the model is unreachable")

    fakes.chat.answer = "fine"
    monkeypatch.setattr(fakes, "chat", explode, raising=False)
    # Rebuild the session so it uses the exploding chat function.
    workflow = build_workflow(deps=fakes.deps(), checkpointer=build_checkpointer(backend="memory"))
    broken = cli.WorkflowSession(workflow, "conv-1")

    _script(monkeypatch, "hello", "exit")
    cli.run_loop(broken)

    out = capsys.readouterr().out
    assert "something went wrong" in out
    assert "RuntimeError" in out
    assert "Bye!" in out  # the session survived and exited normally


def test_a_failed_turn_leaves_the_checkpointed_state_untouched(fakes):
    """No partial state: a turn that raises commits nothing."""
    calls = {"n": 0}

    def flaky(agent, message, thread_id):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the model is unreachable")
        return "ok", ["submit_reservation"]

    fakes.chat = flaky
    workflow = build_workflow(deps=fakes.deps(), checkpointer=build_checkpointer(backend="memory"))
    run_turn(workflow, "conv-1", "yes, confirm")
    before = dict(workflow_state(workflow, "conv-1"))

    with pytest.raises(RuntimeError):
        run_turn(workflow, "conv-1", "and another one")

    after = workflow_state(workflow, "conv-1")
    assert after["booking_id"] == before["booking_id"]
    assert after["approval_status"] == before["approval_status"]


# --- the background ticker ----------------------------------------------------


def test_the_ticker_stops_when_asked(session, monkeypatch):
    monkeypatch.setattr("parking_chatbot.config.ADMIN_NOTIFY_POLL_SECONDS", 0.01)
    stop = threading.Event()
    thread = threading.Thread(target=cli._watch_for_decisions, args=(session, stop), daemon=True)
    thread.start()
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_the_ticker_survives_a_failing_tick(session, monkeypatch):
    monkeypatch.setattr("parking_chatbot.config.ADMIN_NOTIFY_POLL_SECONDS", 0.01)
    tried_twice = threading.Event()
    attempts = {"n": 0}

    def exploding_tick():
        attempts["n"] += 1
        if attempts["n"] >= 2:
            tried_twice.set()
        raise RuntimeError("database is locked")

    monkeypatch.setattr(session, "tick", exploding_tick)
    stop = threading.Event()
    thread = threading.Thread(target=cli._watch_for_decisions, args=(session, stop), daemon=True)
    thread.start()
    try:
        # It kept going after the first failure instead of dying with it.
        assert tried_twice.wait(timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
