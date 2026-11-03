import importlib

import pytest

from parking_chatbot import config

# ADMIN_REQUEST_EXPIRY_MINUTES and ADMIN_NOTIFY_POLL_SECONDS are guarded against
# non-positive values (0 or negative minutes/seconds makes no sense). SQL_BUSY_TIMEOUT_MS
# has no such guard - 0 is SQLite's own normal "don't wait" behavior, a valid setting,
# not an error - so only its happy path is covered here.
_GUARDED_POSITIVE_INT_VARS = ["ADMIN_REQUEST_EXPIRY_MINUTES", "ADMIN_NOTIFY_POLL_SECONDS"]
_POSITIVE_INT_VARS = _GUARDED_POSITIVE_INT_VARS + [
    "SQL_BUSY_TIMEOUT_MS",
    "MCP_RETRY_INTERVAL_SECONDS",
    "MCP_REQUEST_TIMEOUT_SECONDS",
]

# These two have a stricter minimum than "just positive" - checked at their own
# boundary, separately from the generic >0 guards above.
_MINIMUM_INT_VARS = [("MCP_RETRY_INTERVAL_SECONDS", 5), ("MCP_REQUEST_TIMEOUT_SECONDS", 1)]


@pytest.mark.parametrize("var", _GUARDED_POSITIVE_INT_VARS)
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_env_value_raises_on_import(monkeypatch, var, value):
    monkeypatch.setenv(var, str(value))
    try:
        with pytest.raises(ValueError, match=var):
            importlib.reload(config)
    finally:
        monkeypatch.undo()  # restore the real environment...
        importlib.reload(config)  # ...then rebuild config.py from it for later tests


@pytest.mark.parametrize("var", _POSITIVE_INT_VARS)
def test_a_valid_positive_env_value_is_accepted(monkeypatch, var):
    monkeypatch.setenv(var, "42")
    try:
        importlib.reload(config)
        assert getattr(config, var) == 42
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_facility_timezone_defaults_to_utc():
    assert config.FACILITY_TIMEZONE == "UTC"


def test_an_unknown_facility_timezone_raises_on_import(monkeypatch):
    monkeypatch.setenv("FACILITY_TIMEZONE", "Not/A_Real_Zone")
    try:
        with pytest.raises(ValueError, match="FACILITY_TIMEZONE"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_a_valid_facility_timezone_is_accepted(monkeypatch):
    monkeypatch.setenv("FACILITY_TIMEZONE", "Europe/Berlin")
    try:
        importlib.reload(config)
        assert config.FACILITY_TIMEZONE == "Europe/Berlin"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize(("var", "minimum"), _MINIMUM_INT_VARS)
def test_below_minimum_env_value_raises_on_import(monkeypatch, var, minimum):
    monkeypatch.setenv(var, str(minimum - 1))
    try:
        with pytest.raises(ValueError, match=var):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize(("var", "minimum"), _MINIMUM_INT_VARS)
def test_the_minimum_env_value_itself_is_accepted(monkeypatch, var, minimum):
    monkeypatch.setenv(var, str(minimum))
    try:
        importlib.reload(config)
        assert getattr(config, var) == minimum
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_mcp_api_token_defaults_to_none(monkeypatch):
    # Explicitly unset, not just read as-is: CI (and any real .env) may well have
    # this set to something, so asserting the *default* means first guaranteeing
    # it is actually absent, the same way the other reload-based tests here do.
    # load_dotenv() defaults to override=False, so it only fills in variables
    # currently absent from os.environ - exactly the state delenv just created.
    # Without stubbing it out, reload() would silently refill this from a real
    # local .env, defeating the delenv above.
    monkeypatch.delenv("MCP_API_TOKEN", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    try:
        importlib.reload(config)
        assert config.MCP_API_TOKEN is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_mcp_server_url_has_a_working_default(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    try:
        importlib.reload(config)
        assert config.MCP_SERVER_URL == "http://127.0.0.1:8002"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_mcp_reservations_file_path_is_resolved_and_absolute():
    assert config.MCP_RESERVATIONS_FILE_PATH.is_absolute()


def test_mcp_reservations_file_path_defaults_inside_the_project_root():
    assert config.MCP_RESERVATIONS_FILE_PATH == config.PROJECT_ROOT / "data" / "reservations.txt"


def test_default_reservations_file_is_gitignored():
    gitignore = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/reservations.txt" in gitignore


def test_a_relative_mcp_reservations_file_path_is_resolved_against_the_project_root(monkeypatch):
    monkeypatch.setenv("MCP_RESERVATIONS_FILE_PATH", "somewhere/custom.txt")
    try:
        importlib.reload(config)
        assert config.MCP_RESERVATIONS_FILE_PATH == config.PROJECT_ROOT / "somewhere" / "custom.txt"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_absolute_mcp_reservations_file_path_is_used_as_is(monkeypatch, tmp_path):
    absolute = tmp_path / "custom.txt"
    monkeypatch.setenv("MCP_RESERVATIONS_FILE_PATH", str(absolute))
    try:
        importlib.reload(config)
        assert config.MCP_RESERVATIONS_FILE_PATH == absolute
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_static_data_dir_is_inside_the_project_root():
    assert config.STATIC_DATA_DIR == config.PROJECT_ROOT / "data" / "static"


def test_chunk_overlap_is_smaller_than_chunk_size():
    assert 0 < config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_max_advance_days_is_a_positive_integer():
    assert isinstance(config.MAX_ADVANCE_DAYS, int)
    assert config.MAX_ADVANCE_DAYS > 0


def test_top_k_is_a_positive_integer():
    assert isinstance(config.TOP_K, int)
    assert config.TOP_K > 0
