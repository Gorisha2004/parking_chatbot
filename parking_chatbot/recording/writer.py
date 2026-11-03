"""Appending approved-reservation lines to the fixed reservation file.

All file I/O for Stage 3 lives here: the lock around the check-and-append, the
dedup check, and the restrictive permissions. formatting.py only turns data into
text - it never touches disk.
"""
import logging
import os
import stat

from filelock import FileLock

from parking_chatbot import config
from parking_chatbot.recording.formatting import FIELD_SEPARATOR, VehicleLineInput, format_group_lines

logger = logging.getLogger(__name__)


def _lock_path() -> str:
    return f"{config.MCP_RESERVATIONS_FILE_PATH}.lock"


def _already_recorded(booking_id: str) -> bool:
    """Whether this booking's lines are already in the file - found by scanning
    for a matching Pass ID prefix, not by maintaining a separate index that could
    drift out of sync with the file itself. Must only be called while holding the
    lock, so nothing can append between this check and the write below."""
    path = config.MCP_RESERVATIONS_FILE_PATH
    if not path.exists():
        return False
    pass_id_prefix = f"{booking_id}-"
    with path.open("r", encoding="utf-8") as reservations_file:
        for line in reservations_file:
            fields = line.rstrip("\n").split(FIELD_SEPARATOR)
            if len(fields) == 6 and fields[4].startswith(pass_id_prefix):
                return True
    return False


def _ends_without_a_trailing_newline(path) -> bool:
    """True if the file exists, is non-empty, and its last byte isn't a newline -
    e.g. a previous process was killed mid-write. Appending straight onto that would
    silently merge the new line(s) onto the end of that corrupt one."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as reservations_file:
        reservations_file.seek(-1, os.SEEK_END)
        return reservations_file.read(1) != b"\n"


def _set_owner_only_permissions(path) -> None:
    """Owner read/write only. These are POSIX permission bits; Windows ignores
    most of them but still honors the read-only bit, so this tightens things
    everywhere without being a strong guarantee on Windows specifically."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        logger.warning("Could not set restrictive permissions on the reservation file")


def record_group(
    *,
    booking_id: str,
    name: str,
    period_start: str,
    period_end: str,
    approval_time: str,
    vehicles: list[VehicleLineInput],
) -> dict:
    """Append one line per vehicle for this approved booking - the whole group in
    one write, or nothing at all if any field fails validation - or do nothing if
    this booking ID is already recorded.

    Returns {"recorded": True, "lines_written": N}. Raises LineFormatError (from
    formatting.py) if the data is invalid; raises OSError for a genuine I/O
    failure (disk full, permission denied, ...) - both are the caller's (the MCP
    tool's) job to turn into a clear response, not this function's.
    """
    path = config.MCP_RESERVATIONS_FILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(_lock_path()):
        if _already_recorded(booking_id):
            logger.info("Booking %s already recorded; nothing written", booking_id)
            return {"recorded": True, "lines_written": 0}

        lines = format_group_lines(
            booking_id=booking_id,
            name=name,
            period_start=period_start,
            period_end=period_end,
            approval_time=approval_time,
            vehicles=vehicles,
        )

        file_existed = path.exists()
        leading_newline = "\n" if _ends_without_a_trailing_newline(path) else ""
        with path.open("a", encoding="utf-8") as reservations_file:
            reservations_file.write(leading_newline + "\n".join(lines) + "\n")
        if not file_existed:
            _set_owner_only_permissions(path)

    logger.info("Recorded booking %s: %d line(s)", booking_id, len(lines))
    return {"recorded": True, "lines_written": len(lines)}
