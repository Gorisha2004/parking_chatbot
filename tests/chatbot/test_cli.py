import pytest

from parking_chatbot.chatbot import cli


@pytest.fixture
def run_cli(monkeypatch, capsys):
    """Run cli.main() with scripted user input and a fake chat function; return stdout."""

    def _run(inputs, chat_fn, argv=("prog",)):
        lines = iter(inputs)

        def fake_input(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr("sys.argv", list(argv))
        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(cli, "build_agent", lambda: object())
        monkeypatch.setattr(cli, "chat", chat_fn)
        cli.main()
        return capsys.readouterr().out

    return _run


def test_parse_args_debug_flag_defaults_to_false(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    assert cli.parse_args().debug is False


def test_parse_args_debug_flag_can_be_enabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--debug"])
    assert cli.parse_args().debug is True


def test_main_prints_the_bot_answer_and_exits_on_quit(run_cli):
    out = run_cli(["What are your prices?", "quit"], lambda agent, msg, tid: ("Hourly is 3 USD.", []))
    assert "Bot: Hourly is 3 USD." in out
    assert "Bye!" in out


def test_main_skips_empty_input_without_calling_the_agent(run_cli):
    calls = []
    run_cli(["", "   ", "exit"], lambda agent, msg, tid: calls.append(msg) or ("x", []))
    assert calls == []


def test_main_keeps_one_thread_id_for_the_whole_session(run_cli):
    thread_ids = []
    run_cli(["first", "second", "exit"], lambda agent, msg, tid: thread_ids.append(tid) or ("ok", []))
    assert len(thread_ids) == 2
    assert thread_ids[0] == thread_ids[1]


def test_main_recovers_from_an_agent_error_and_keeps_running(run_cli):
    def failing_then_ok(agent, msg, tid):
        if msg == "boom":
            raise RuntimeError("LLM unavailable")
        return ("still here", [])

    out = run_cli(["boom", "hello", "exit"], failing_then_ok)

    assert "Sorry, something went wrong (RuntimeError)" in out
    assert "Bot: still here" in out


def test_main_shows_tools_only_in_debug_mode(run_cli):
    chat_fn = lambda agent, msg, tid: ("Level 2 has space.", ["lookup_availability"])  # noqa: E731
    assert "lookup_availability" not in run_cli(["space?", "exit"], chat_fn)
    assert "lookup_availability" in run_cli(["space?", "exit"], chat_fn, argv=("prog", "--debug"))


def test_main_exits_cleanly_on_end_of_input(run_cli):
    out = run_cli([], lambda agent, msg, tid: ("unused", []))
    assert "Bye!" in out


def test_decision_message_for_an_approval_says_congrats():
    message = cli._decision_message({"booking_id": "PK-ABC123", "status": "approved", "decision_reason": None})
    assert message == "Congrats! Your booking PK-ABC123 has been approved."


def test_decision_message_for_a_refusal_includes_the_reason():
    message = cli._decision_message({"booking_id": "PK-ABC123", "status": "refused", "decision_reason": "no space"})
    assert message == "Your booking PK-ABC123 was refused. Reason: no space"


def test_decision_message_for_an_expiry_says_so():
    message = cli._decision_message({"booking_id": "PK-ABC123", "status": "expired", "decision_reason": None})
    assert "expired" in message


def test_announce_decisions_prints_one_message_per_decided_booking(monkeypatch, capsys):
    bookings = [
        {"booking_id": "PK-AAA111", "status": "approved", "decision_reason": None},
        {"booking_id": "PK-BBB222", "status": "refused", "decision_reason": None},
    ]
    monkeypatch.setattr(cli, "pending_decisions", lambda thread_id: bookings)

    cli._announce_decisions("some-thread")

    out = capsys.readouterr().out
    assert "Congrats! Your booking PK-AAA111 has been approved." in out
    assert "Your booking PK-BBB222 was refused." in out


def test_announce_decisions_prints_nothing_when_there_is_no_decision_yet(monkeypatch, capsys):
    monkeypatch.setattr(cli, "pending_decisions", lambda thread_id: [])

    cli._announce_decisions("some-thread")

    assert capsys.readouterr().out == ""


def test_watch_for_decisions_polls_until_told_to_stop(monkeypatch):
    import threading

    calls = []
    stop_event = threading.Event()

    def fake_announce(thread_id):
        calls.append(thread_id)
        stop_event.set()  # stop after the first poll, so the test doesn't hang

    monkeypatch.setattr(cli, "_announce_decisions", fake_announce)
    monkeypatch.setattr(cli.config, "ADMIN_NOTIFY_POLL_SECONDS", 0)

    cli._watch_for_decisions("some-thread", stop_event)

    assert calls == ["some-thread"]
