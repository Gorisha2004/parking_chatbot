import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from parking_chatbot import config


def get_connection() -> sqlite3.Connection:
    """Open the app database with foreign-key enforcement (SQLite has it off by default).

    Three separate processes can now have this database open at once (the chatbot,
    the admin chat console, the admin REST server): WAL mode lets readers proceed
    without waiting on a writer, and an explicit busy_timeout makes a writer that does
    collide with another wait and retry instead of raising "database is locked"
    immediately. (Python's sqlite3 module already defaults to a 5s busy timeout, but
    setting it explicitly here makes it a property of this app, not an implicit
    library default.)
    """
    conn = sqlite3.connect(config.SQL_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {config.SQL_BUSY_TIMEOUT_MS}")
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
