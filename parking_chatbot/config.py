import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()
DIAL_ENDPOINT = os.getenv("DIAL_ENDPOINT")
DIAL_API_KEY = os.getenv("DIAL_API_KEY")
DIAL_API_VERSION = os.getenv("DIAL_API_VERSION")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DATA_DIR = PROJECT_ROOT / "data" / "static"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "parking_static_docs"
TOP_K = 5
SQL_DB_PATH = "data/parking.db"
# How long a connection waits for a lock held by another process/thread before
# raising "database is locked", instead of failing immediately.
SQL_BUSY_TIMEOUT_MS = int(os.getenv("SQL_BUSY_TIMEOUT_MS", "5000"))
MAX_ADVANCE_DAYS = int(os.getenv("MAX_ADVANCE_DAYS", "90"))
MAX_VEHICLES_PER_BOOKING = int(os.getenv("MAX_VEHICLES_PER_BOOKING", "10"))
# Pass IDs end in a two-digit vehicle number (PK-XXXXXX-01), so a booking can never exceed 99.
if not 1 <= MAX_VEHICLES_PER_BOOKING <= 99:
    raise ValueError("MAX_VEHICLES_PER_BOOKING must be between 1 and 99.")
# How long a booking waits for an administrator decision before it lazily expires
# (checked whenever a booking is read, not by a background job).
ADMIN_REQUEST_EXPIRY_MINUTES = int(os.getenv("ADMIN_REQUEST_EXPIRY_MINUTES", "60"))
if ADMIN_REQUEST_EXPIRY_MINUTES <= 0:
    raise ValueError("ADMIN_REQUEST_EXPIRY_MINUTES must be positive.")
# Shared bearer token the admin server requires on every request. No default: if it
# is not set, every request is rejected (fail closed) rather than left open.
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")
# Where scripts/run_admin_server.py listens; scripts/admin_cli.py talks to it here.
ADMIN_SERVER_URL = os.getenv("ADMIN_SERVER_URL", "http://127.0.0.1:8001")
# How often the chatbot CLI checks, in the background, whether an administrator has
# decided on a booking submitted in the current conversation.
ADMIN_NOTIFY_POLL_SECONDS = int(os.getenv("ADMIN_NOTIFY_POLL_SECONDS", "3"))
if ADMIN_NOTIFY_POLL_SECONDS <= 0:
    raise ValueError("ADMIN_NOTIFY_POLL_SECONDS must be positive.")
# start_time, end_time and decided_at are all stored naive, in the facility's own
# wall-clock time (never converted at write time - see reservation/models.py and
# db/admin.py). Stage 3 is the first thing that needs an unambiguous UTC instant
# (for the reservation file), so this is the one place that time zone is named -
# an IANA zone, e.g. "Europe/Berlin". Default "UTC" means "treat the naive values
# as already UTC" - a safe no-op conversion, not a guess at the facility's real zone.
FACILITY_TIMEZONE = os.getenv("FACILITY_TIMEZONE", "UTC")
try:
    ZoneInfo(FACILITY_TIMEZONE)
except ZoneInfoNotFoundError as error:
    raise ValueError(f"FACILITY_TIMEZONE {FACILITY_TIMEZONE!r} is not a known IANA time zone name.") from error

# --- Stage 3: recording approved reservations to a file via the MCP server --------------------

# Where scripts/run_mcp_server.py listens; the recording client (parking_chatbot.recording)
# talks to it here.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8002")
# Bearer token the recording server requires on every request. No default: if it is
# not set (or set to an empty string), the server refuses to start and the client
# refuses to send - fail closed, like ADMIN_API_TOKEN, but stricter: this one isn't
# allowed to come up at all in a half-configured state. Enforced where this is
# actually used (the server's startup, the client's send function), not here - this
# module is imported by everything, including code that has nothing to do with
# Stage 3, so it must not force every process to have this token set just to start.
MCP_API_TOKEN = os.getenv("MCP_API_TOKEN")
# The reservation file's path, resolved once, here, against the project root - not a
# bare relative string like SQL_DB_PATH, which means whatever the process's current
# directory happens to be. A relative value is relative to the project root
# regardless of where the command was run from; an absolute value is used as-is.
MCP_RESERVATIONS_FILE_PATH = (PROJECT_ROOT / os.getenv("MCP_RESERVATIONS_FILE_PATH", "data/reservations.txt")).resolve()
# How often the background sweep retries approved bookings the server hasn't
# confirmed recording yet. At least 5s: anything shorter is really a busy-loop, not
# a retry interval.
MCP_RETRY_INTERVAL_SECONDS = int(os.getenv("MCP_RETRY_INTERVAL_SECONDS", "30"))
if MCP_RETRY_INTERVAL_SECONDS < 5:
    raise ValueError("MCP_RETRY_INTERVAL_SECONDS must be at least 5 seconds.")
# How long the recording client waits for one HTTP call to the server before giving
# up. At least 1s: anything shorter is unreliable even for a local network call.
MCP_REQUEST_TIMEOUT_SECONDS = int(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "5"))
if MCP_REQUEST_TIMEOUT_SECONDS < 1:
    raise ValueError("MCP_REQUEST_TIMEOUT_SECONDS must be at least 1 second.")

# Path is Python’s modern way to work with file paths (it handles / vs \ for Windows/Mac automatically).
# __file__ is a built-in variable holding the path of the current file (config.py).
# .resolve() turns it into a full absolute path.
# .parent goes one folder up. Doing it twice goes from config.py → parting_chatbot/ → the project root.
# / "data" / "static" joins path parts (yes, / is overloaded in Path to mean “join”).

# --- Stage 4: LangGraph orchestration of the whole pipeline -----------------------------------

# Which LangGraph checkpointer the workflow uses to persist its shared state.
# "sqlite" (the default) keeps a pending approval across process restarts, which is
# the point of checkpointing here; "memory" keeps it only for the life of the
# process and is what the test suite and the load harness use, so neither leaves a
# file behind. Anything else is a configuration error, not a silent fallback.
WORKFLOW_CHECKPOINT_BACKEND = os.getenv("WORKFLOW_CHECKPOINT_BACKEND", "sqlite").strip().lower()
if WORKFLOW_CHECKPOINT_BACKEND not in ("sqlite", "memory"):
    raise ValueError('WORKFLOW_CHECKPOINT_BACKEND must be either "sqlite" or "memory".')
# Where the sqlite checkpointer stores workflow state. Resolved against the project
# root like MCP_RESERVATIONS_FILE_PATH, so it does not depend on the current directory.
WORKFLOW_CHECKPOINT_DB_PATH = (
    PROJECT_ROOT / os.getenv("WORKFLOW_CHECKPOINT_DB_PATH", "data/workflow_checkpoints.sqlite")
).resolve()
