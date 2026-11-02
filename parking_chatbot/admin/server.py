"""The admin REST API: the channel through which an administrator reviews and
decides on escalated bookings.

Deliberately thin: all the actual logic (rendering, reply interpretation, decision
storage) lives in admin/agent.py and db/admin.py. This module only wires HTTP to it.
"""
from contextlib import asynccontextmanager
from secrets import compare_digest

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from parking_chatbot import config
from parking_chatbot.admin.agent import AmbiguousReplyError, interpret_and_apply, render_request
from parking_chatbot.db import admin as admin_db
from parking_chatbot.db.schema import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Running this server is the only thing that needs the admin-decision columns,
    # but nothing else in the project calls create_tables() before it - so the
    # server must do it itself on startup. Safe to run every time: it only adds
    # missing tables/columns, it never drops or overwrites existing data.
    create_tables()
    yield


app = FastAPI(title="Parking Chatbot - Admin API", lifespan=lifespan)


# auto_error=False: FastAPI's default behavior raises 403 for a missing/malformed
# header. We want 401 in every failure case (missing, malformed, wrong or unset),
# so errors are raised by require_admin_token below instead. Using this as a
# dependency (rather than reading the header by hand) is what makes /docs show an
# Authorize button and attach the token to requests made from there.
bearer_scheme = HTTPBearer(auto_error=False)


def require_admin_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> None:
    """Fail closed: if no token is configured, nothing can ever authenticate."""
    expected = config.ADMIN_API_TOKEN
    provided = credentials.credentials if credentials else None
    if not expected or not provided or not compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token")


class DecisionRequest(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/requests", dependencies=[Depends(require_admin_token)])
def list_requests() -> list[dict]:
    """Every booking currently awaiting a decision, with a human-readable summary."""
    return [
        {"request": request.model_dump(mode="json"), "summary": render_request(request)}
        for request in admin_db.list_pending_bookings()
    ]


@app.post("/requests/{booking_id}/decision", dependencies=[Depends(require_admin_token)])
def submit_decision(booking_id: str, body: DecisionRequest) -> dict:
    """Interpret the administrator's free-text reply and record the decision - the
    same interpret_and_apply() path the admin chat agent's record_decision tool uses,
    so there is exactly one place a decision is ever written."""
    try:
        result = interpret_and_apply(booking_id, body.reply)
    except AmbiguousReplyError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except admin_db.UnknownBookingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except admin_db.AlreadyDecidedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    return {"booking_id": result.booking_id, "decision": result.decision, "reason": result.reason}
