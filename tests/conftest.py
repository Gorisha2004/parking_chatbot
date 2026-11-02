"""Shared pytest fixtures for the whole test suite.

External services (LLM, embeddings, vector store, network) are never hit in
these tests: they are replaced with fakes or an isolated temporary SQLite
database.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from parking_chatbot import config
from parking_chatbot.db.schema import create_tables


@pytest.fixture(autouse=True)
def _isolate_sql_db_path(tmp_path, monkeypatch):
    """Point every test at its own temporary SQLite path - autouse, so no test can
    read or write the real data/parking.db by accident, even one that doesn't
    explicitly ask for the database.

    Deliberately does NOT create the schema: a test that touches the database
    without declaring the sql_db fixture below should fail loudly, here, in any
    environment - not pass by accident against a developer's real local database
    (which has tables) and then fail in CI (which starts with none). That exact gap
    is what let two tests pass locally and fail in the pipeline.
    """
    db_path = tmp_path / "test_parking.db"
    monkeypatch.setattr(config, "SQL_DB_PATH", str(db_path))
    return db_path


@pytest.fixture
def sql_db(_isolate_sql_db_path):
    """Create the schema in the current test's isolated temporary database. Depend on
    this whenever a test actually reads or writes the database, or needs the ordering
    guarantee that the schema exists before something else (e.g. a FastAPI TestClient's
    startup lifespan) touches it."""
    create_tables()
    return _isolate_sql_db_path


@dataclass
class FakeMessage:
    """Minimal stand-in for a LangChain message."""

    type: str
    content: str
    name: str | None = None


@dataclass
class FakeChatModel:
    """Minimal stand-in for a BaseChatModel: records calls, returns a fixed reply."""

    reply: str = "fake response"
    calls: list = field(default_factory=list)

    def invoke(self, prompt):
        self.calls.append(prompt)
        return FakeMessage(type="ai", content=self.reply)


@pytest.fixture
def fake_chat_model():
    return FakeChatModel()
