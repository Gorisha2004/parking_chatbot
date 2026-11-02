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


@pytest.fixture
def sql_db(tmp_path, monkeypatch):
    """Point the app at a fresh, temporary SQLite database for one test."""
    db_path = tmp_path / "test_parking.db"
    monkeypatch.setattr(config, "SQL_DB_PATH", str(db_path))
    create_tables()
    return db_path


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
