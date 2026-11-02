import importlib

import pytest

from parking_chatbot import config

# ADMIN_REQUEST_EXPIRY_MINUTES and ADMIN_NOTIFY_POLL_SECONDS are guarded against
# non-positive values (0 or negative minutes/seconds makes no sense). SQL_BUSY_TIMEOUT_MS
# has no such guard - 0 is SQLite's own normal "don't wait" behavior, a valid setting,
# not an error - so only its happy path is covered here.
_GUARDED_POSITIVE_INT_VARS = ["ADMIN_REQUEST_EXPIRY_MINUTES", "ADMIN_NOTIFY_POLL_SECONDS"]
_POSITIVE_INT_VARS = _GUARDED_POSITIVE_INT_VARS + ["SQL_BUSY_TIMEOUT_MS"]


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
