import os
from pathlib import Path
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

# Path is Python’s modern way to work with file paths (it handles / vs \ for Windows/Mac automatically).
# __file__ is a built-in variable holding the path of the current file (config.py).
# .resolve() turns it into a full absolute path.
# .parent goes one folder up. Doing it twice goes from config.py → parting_chatbot/ → the project root.
# / "data" / "static" joins path parts (yes, / is overloaded in Path to mean “join”).