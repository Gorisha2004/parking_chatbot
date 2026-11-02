import importlib.util

import pytest

from parking_chatbot import config


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, config.PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admin_cli = _load_script("admin_cli")


@pytest.fixture
def run_cli(monkeypatch, capsys):
    """Run admin_cli.main() with scripted admin input and a fake chat function; return stdout."""

    def _run(inputs, chat_fn, argv=("prog",)):
        lines = iter(inputs)

        def fake_input(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr("sys.argv", list(argv))
        monkeypatch.setattr("builtins.input", fake_input)
        monkeypatch.setattr(admin_cli, "build_admin_agent", lambda: object())
        monkeypatch.setattr(admin_cli, "admin_chat", chat_fn)
        admin_cli.main()
        return capsys.readouterr().out

    return _run


def test_parse_args_debug_flag_defaults_to_false(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    assert admin_cli.parse_args().debug is False


def test_parse_args_debug_flag_can_be_enabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--debug"])
    assert admin_cli.parse_args().debug is True


def test_main_prints_the_agent_reply_and_exits_on_q(run_cli):
    out = run_cli(["approve PK-ABCDEF", "q"], lambda agent, msg, tid: ("You're about to APPROVE PK-ABCDEF.", []))
    assert "Bot: You're about to APPROVE PK-ABCDEF." in out
    assert "Bye!" in out


@pytest.mark.parametrize("exit_word", ["exit", "quit", "q", "Q"])
def test_main_quits_on_any_exit_command(run_cli, exit_word):
    out = run_cli([exit_word], lambda agent, msg, tid: pytest.fail("must not call the agent on an exit command"))
    assert "Bye!" in out


def test_main_skips_empty_input_without_calling_the_agent(run_cli):
    calls = []
    run_cli(["", "   ", "q"], lambda agent, msg, tid: calls.append(msg) or ("x", []))
    assert calls == []


def test_main_keeps_one_thread_id_for_the_whole_session(run_cli):
    thread_ids = []
    run_cli(["first", "second", "q"], lambda agent, msg, tid: thread_ids.append(tid) or ("ok", []))
    assert len(thread_ids) == 2
    assert thread_ids[0] == thread_ids[1]


def test_main_recovers_from_an_agent_error_and_keeps_running(run_cli):
    def failing_then_ok(agent, msg, tid):
        if msg == "boom":
            raise RuntimeError("LLM unavailable")
        return ("still here", [])

    out = run_cli(["boom", "hello", "q"], failing_then_ok)

    assert "Sorry, something went wrong (RuntimeError)" in out
    assert "Bot: still here" in out


def test_main_shows_tools_only_in_debug_mode(run_cli):
    chat_fn = lambda agent, msg, tid: ("1 pending.", ["list_pending_requests"])  # noqa: E731
    assert "list_pending_requests" not in run_cli(["list", "q"], chat_fn)
    assert "list_pending_requests" in run_cli(["list", "q"], chat_fn, argv=("prog", "--debug"))


def test_main_exits_cleanly_on_end_of_input(run_cli):
    out = run_cli([], lambda agent, msg, tid: ("unused", []))
    assert "Bye!" in out


def test_main_exits_cleanly_on_ctrl_c(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(admin_cli, "build_admin_agent", lambda: object())

    def raise_interrupt(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raise_interrupt)
    admin_cli.main()
    assert "Bye!" in capsys.readouterr().out
