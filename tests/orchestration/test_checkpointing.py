"""Tests for choosing and persisting with a checkpoint backend."""
import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from parking_chatbot import config
from parking_chatbot.orchestration.checkpointing import build_checkpointer, close_checkpointer


def test_the_memory_backend_gives_an_in_memory_saver():
    checkpointer = build_checkpointer(backend="memory")
    assert isinstance(checkpointer, InMemorySaver)
    close_checkpointer(checkpointer)  # a no-op, but must not raise


def test_the_sqlite_backend_creates_its_database_and_tables(tmp_path):
    path = tmp_path / "nested" / "checkpoints.sqlite"
    checkpointer = build_checkpointer(backend="sqlite", db_path=path)
    try:
        assert isinstance(checkpointer, SqliteSaver)
        assert path.exists()
        tables = {row[0] for row in checkpointer.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "checkpoints" in tables
    finally:
        close_checkpointer(checkpointer)


def test_the_backend_defaults_to_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKFLOW_CHECKPOINT_BACKEND", "memory")
    assert isinstance(build_checkpointer(), InMemorySaver)


def test_an_unknown_backend_fails_loudly_instead_of_forgetting_silently():
    with pytest.raises(ValueError, match="Unknown workflow checkpoint backend"):
        build_checkpointer(backend="redis")


def test_close_checkpointer_releases_the_sqlite_connection(tmp_path):
    checkpointer = build_checkpointer(backend="sqlite", db_path=tmp_path / "c.sqlite")
    close_checkpointer(checkpointer)
    with pytest.raises(sqlite3.ProgrammingError):
        checkpointer.conn.execute("SELECT 1")
