import importlib.util

import pytest
import uvicorn

from parking_chatbot import config


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, config.PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loading_the_module_does_not_start_a_server(monkeypatch):
    """The `if __name__ == "__main__":` guard must mean loading this file (as every
    test, and `import`, does) never actually calls uvicorn.run."""
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("must not run a server just by loading the module"))
    _load_script("run_admin_server")


def test_main_runs_the_admin_app_on_the_expected_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))

    run_admin_server = _load_script("run_admin_server")
    run_admin_server.main()

    [(args, kwargs)] = calls
    assert args == ("parking_chatbot.admin.server:app",)
    assert kwargs == {"host": "127.0.0.1", "port": 8001}
