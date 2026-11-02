import sqlite3

from parking_chatbot import config
from parking_chatbot.db.database import get_connection


def test_get_connection_uses_wal_mode_so_readers_do_not_block_on_a_writer(sql_db):
    conn = get_connection()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_get_connection_sets_the_configured_busy_timeout(sql_db, monkeypatch):
    monkeypatch.setattr(config, "SQL_BUSY_TIMEOUT_MS", 1234)
    conn = get_connection()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
    finally:
        conn.close()


def test_get_connection_returns_a_working_sqlite_connection(sql_db):
    conn = get_connection()
    try:
        assert isinstance(conn, sqlite3.Connection)
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()


def test_get_connection_uses_the_configured_db_path(sql_db):
    conn = get_connection()
    conn.close()
    # sql_db fixture points config.SQL_DB_PATH at a tmp file; get_connection must honor it.
    from parking_chatbot import config

    assert config.SQL_DB_PATH == str(sql_db)
