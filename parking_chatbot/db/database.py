import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from parking_chatbot import config


def get_connection() -> sqlite3.Connection:
    """Open the app database with foreign-key enforcement (SQLite has it off by default)."""
    conn = sqlite3.connect(config.SQL_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def immediate_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block as one transaction that takes the write lock up front.

    Two writers can then never interleave (check-then-insert stays consistent), and
    everything inside, including table changes, is rolled back if the block fails.
    """
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.isolation_level = ""
