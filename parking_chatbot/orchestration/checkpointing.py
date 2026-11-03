"""Where the workflow's shared state is persisted between turns.

The graph is invoked once per turn and ends the turn while a booking is still
pending, so the checkpointer is what remembers that booking. With the sqlite
backend that memory also survives the process: the administrator can take as
long as they like, the chatbot can be restarted in the meantime, and the
conversation still knows which booking it is waiting on.

The backend is chosen by configuration, not by code, so the test suite and the
load harness can run entirely in memory and leave no file behind.
"""
import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from parking_chatbot import config

logger = logging.getLogger(__name__)


def build_checkpointer(
    backend: str | None = None,
    db_path: str | Path | None = None,
) -> BaseCheckpointSaver:
    """The checkpointer the workflow should use.

    Arguments override configuration, which is what lets a test ask for a
    specific backend and path without touching the environment.

    Raises:
        ValueError: if the backend name is not one this project supports. A typo
            must fail loudly rather than silently fall back to a backend that
            forgets everything on exit.
    """
    backend = (backend or config.WORKFLOW_CHECKPOINT_BACKEND).strip().lower()
    if backend == "memory":
        logger.debug("Using the in-memory checkpointer; workflow state will not survive this process")
        return InMemorySaver()
    if backend != "sqlite":
        raise ValueError(f'Unknown workflow checkpoint backend {backend!r}: expected "sqlite" or "memory".')

    path = Path(db_path) if db_path is not None else config.WORKFLOW_CHECKPOINT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the CLI drives the graph from the input loop and
    # from the background decision ticker, and those are different threads. The
    # two are serialized by a lock in the CLI, so the connection is still only
    # ever used by one of them at a time.
    connection = sqlite3.connect(path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()  # create the checkpoint tables if this is a new file
    logger.debug("Using the sqlite checkpointer at %s", path)
    return checkpointer


def close_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """Release the checkpointer's resources, if it holds any.

    In-memory checkpointers hold none, so this is a no-op for them - callers can
    always call it in a `finally` without caring which backend they were given.
    """
    connection = getattr(checkpointer, "conn", None)
    if isinstance(connection, sqlite3.Connection):
        connection.close()
