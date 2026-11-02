# CityPark Central: Parking Reservation Chatbot (CLI)

A command-line chatbot for the CityPark Central parking facility. It answers questions using Retrieval-Augmented Generation (RAG), reads live data (opening hours, prices, free spaces) from a SQL database, collects reservation details through a validated dialogue, escalates every booking to a human administrator for approval, and applies guardrails so sensitive data never reaches the user.

The chat model and embedding model are **EPAM DIAL** (an Azure OpenAI-compatible proxy). Both are built behind LangChain's abstract interfaces (`BaseChatModel`, `Embeddings`), so swapping provider only means editing `llm.py` / `embeddings.py` - no other file depends on the concrete class.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Configuration](#configuration)
5. [Usage walkthrough](#usage-walkthrough)
6. [Stage 2: human-in-the-loop approval](#stage-2-human-in-the-loop-approval)
7. [Design details](#design-details)
8. [Guardrails](#guardrails)
9. [Testing](#testing)
10. [Evaluation](#evaluation)
11. [Project structure](#project-structure)
12. [Known limitations](#known-limitations)

---

## Features

- **Information answers (RAG):** general info, location, facilities, allowed vehicles, rules, payment, contacts, FAQ and the booking process, answered from documents in a Chroma vector database.
- **Live data (SQL):** working hours, prices per vehicle type, and free spaces come from SQLite, never from embeddings.
- **Reservation dialogue:** collects first name, last name, car number, vehicle type and reservation period, validates each value, and supports corrections, cancelling and group bookings.
- **Human-in-the-loop approval:** every submitted booking is escalated to an administrator, who reviews and decides (approve/refuse, with a reason) through a REST API, a Swagger UI, or an interactive console - the user can check the outcome by booking ID at any time.
- **Guardrails:** blocks prompt injection and fishing for other customers' data, and detects and masks sensitive data in both directions using a pretrained NER model (Presidio).
- **Evaluation:** Precision@K, Recall@K, latency (mean, p95, max), tool-routing accuracy and honesty on unanswerable questions, reproducible from one command.

---

## Architecture

```mermaid
flowchart LR
    U["User (CLI)"] --> IG["Input guardrails"]
    IG --> A["Chat agent (LangChain tool-calling)"]
    A --> T1["search_parking_info (RAG)"]
    A --> T2["lookup_hours / prices / availability"]
    A --> T3["reservation tools"]
    A --> T4["check_booking_status"]
    T1 --> C[("Chroma vector DB")]
    T2 --> S[("SQLite, WAL mode")]
    T3 --> S
    T4 --> S
    T3 -->|"on submit"| CH["AdminChannel"]
    CH --> S
    A --> OG["Output guardrails"]
    OG --> U

    Admin["Administrator"] --> ACLI["admin_cli.py (chat)"]
    Admin --> Docs["Swagger UI (/docs)"]
    ACLI --> DET["Deterministic confirmation flow"]
    DET -->|"propose / confirm / cancel"| S
    DET -->|"plain browsing"| AA["Admin agent (LangChain tool-calling)"]
    AA --> AT1["list_pending_requests"]
    AA --> AT2["get_request_details"]
    AA --> AT3["record_decision"]
    AT1 --> S
    AT2 --> S
    AT3 --> S
    Docs --> API["Admin REST API"]
    API --> S
```

Three independent processes can share the same SQLite database: the chatbot (`uv run parking-bot`), the admin console (`uv run python scripts/admin_cli.py`), and the admin REST server (`uv run python scripts/run_admin_server.py`, used by Swagger/`curl`/automation). None depends on another being up - submitting a booking is a local database write, not a network call, and WAL mode plus an explicit busy timeout (`get_connection()` in `db/database.py`) let them read and write concurrently without one blocking another into a "database is locked" error.

There are two LangChain tool-calling agents in this project, not one:

| | Chat agent (`chatbot/agent.py`) | Admin agent (`admin/agent.py`) |
|---|---|---|
| Talks to | the customer | the administrator |
| Tools | `search_parking_info`, `lookup_*`, reservation tools, `check_booking_status` | `list_pending_requests`, `get_request_details`, `record_decision` |
| Writes a decision? | No - only escalates | Only as a defense-in-depth fallback - see below |

The admin agent's tools exist for browsing (listing pending bookings, showing one in full) and open-ended conversation. Deciding on a booking - the safety-critical part - is deliberately **not** left to the LLM: it is handled by plain code (`update_proposal`/`apply_decision` in `admin/agent.py`) that runs on the administrator's raw message before the agent is even invoked. See [Interpreting the administrator's reply](#interpreting-the-administrators-reply) for exactly how.

### Data split: static vs dynamic

| Data | Store | Why |
|---|---|---|
| General info, location, facilities, allowed vehicles, rules, payment, contacts, FAQ, booking process (`data/static/*.md`) | **Chroma** vector DB | Free text, rarely changes, needs semantic search |
| Working hours, prices (per vehicle type and plan), free spaces, reservations and their approval status | **SQLite** (`data/parking.db`) | Structured, changes often, must be exact; never answered from embeddings |

Prices depend on vehicle type, because bigger vehicles take more space: `motorcycle` (two-wheelers) < `three_wheeler` < `car` (cars and SUVs) < `van` (small vans and minibuses up to 2.1 m). Demo prices are defined in `scripts/seed_db.py`.

---

## Quick start

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Access to an EPAM DIAL endpoint with a chat deployment and an embedding deployment

### Install

```bash
# 1. Install dependencies (creates .venv)
uv sync

# 2. Create your configuration
cp .env.example .env        # Windows PowerShell: copy .env.example .env
#    then edit .env (see Configuration below)
```

### Load the data

Both steps are idempotent and safe to re-run; an older database is upgraded in place, never dropped.

```bash
# Create SQL tables and load demo hours / prices / availability.
uv run python scripts/seed_db.py

# Chunk the static docs in data/static/, embed them and store them in Chroma
uv run python scripts/ingest.py
```

![Ingestion output](screenshots/stage1/02_ingest.png)

To inspect what was stored (SQL tables, and Chroma collection with sample chunks):

![Database and vector store contents](screenshots/stage1/03_data_inspection.png)

### Run the chatbot

```bash
uv run parking-bot            # or: uv run python -m parking_chatbot.chatbot.cli
uv run parking-bot --debug    # also shows tools used, latency and error tracebacks
```

Type `exit` or `quit` to leave.

![Chatbot start](screenshots/stage1/04_start_bot.png)

### Run the admin side (Stage 2)

In a second terminal, so the administrator can review and decide on bookings the chatbot escalates - no admin server needs to be running for this, it talks to the database directly:

```bash
uv run python scripts/admin_cli.py   # conversational console for the administrator
```

The REST API (`uv run python scripts/run_admin_server.py`, on `http://127.0.0.1:8001`) is a separate, optional way to reach the same decisions - for Swagger UI (`/docs`) or scripted/automated use.

See [Stage 2: human-in-the-loop approval](#stage-2-human-in-the-loop-approval) for the full walkthrough.

---

## Configuration

All settings come from environment variables (`.env`).

| Variable | Required | Description |
|---|---|---|
| `DIAL_ENDPOINT` | yes | Base URL of the DIAL endpoint |
| `DIAL_API_KEY` | yes | API key for that endpoint |
| `DIAL_API_VERSION` | yes | API version, for example `2024-02-01` |
| `CHAT_MODEL` | yes | Chat deployment name (must support tool/function calling) |
| `EMBEDDING_MODEL` | yes | Embedding deployment name |
| `MAX_ADVANCE_DAYS` | optional | How many days ahead a reservation may start (default `90`) |
| `MAX_VEHICLES_PER_BOOKING` | optional | Largest group booking, 1 to 99 (default `10`) |
| `ADMIN_API_TOKEN` | yes, for the admin server | Bearer token the admin server requires on every request. No default: if unset, every request is rejected rather than left open. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ADMIN_SERVER_URL` | optional | Where the admin REST server itself listens (default `http://127.0.0.1:8001`); not used by `admin_cli.py`, which talks to the admin agent directly, in-process |
| `ADMIN_REQUEST_EXPIRY_MINUTES` | optional | How long a booking waits for a decision before it lazily expires (default `60`) |
| `ADMIN_NOTIFY_POLL_SECONDS` | optional | How often the chatbot CLI checks in the background for a decision on a booking submitted in the current conversation (default `3`) |
| `SQL_BUSY_TIMEOUT_MS` | optional | How long a database connection waits for a lock held by another process before giving up (default `5000`) - relevant since the chatbot, admin console and admin server can all have the database open at once |

> **Important:** if you change `EMBEDDING_MODEL`, re-run `uv run python scripts/ingest.py` - different models produce vectors of different sizes and meaning, so the existing vector store must be rebuilt. Never commit `.env`; only `.env.example` with placeholders belongs in git.

---

## Usage walkthrough

Actual wording comes from the LLM and will differ between runs.

### Asking for information

Working hours and prices come from SQL; facilities and rules come from the documents.

![Hours and prices](screenshots/stage1/05_info_hours_prices.png)

![Location and facilities](screenshots/stage1/06_info_location_facilities.png)

Availability is read live from the database for the requested period.

![Availability](screenshots/stage1/07_info_availability.png)

### Out-of-scope questions

When the documents do not contain the answer, the bot says so and refers the user to customer support instead of inventing something.

![Unanswerable question](screenshots/stage1/08_unanswerable.png)

### Debug mode

With `--debug`, the CLI shows which tool the agent called, what was retrieved, and how long the turn took.

![Debug output](screenshots/stage1/09_debug_retrieval.png)

### Making a reservation

The bot collects missing details, validates every value (invalid plate, past date or unknown vehicle type is rejected with a reason), shows a summary, and saves the booking only after an explicit "yes". On submission the user receives a booking ID and a pass ID per vehicle, and the booking is escalated to the administrator.

![Booking flow, validation and confirmation](screenshots/stage1/10_booking_flow.png)

The saved booking, with status `pending`:

![Booking stored in the database](screenshots/stage1/13_booking_in_db.png)

### Getting notified automatically when a decision is made

The CLI doesn't make the user ask: a background check, running for the life of the session, watches every booking submitted in that conversation and announces a decision the moment it sees one - no need to ask "is it approved?" first:

```
You: book a space for tomorrow 9 to 12, Anna Smith, car AB123CD
Bot: Your booking PK-EKSG9Z is submitted and pending administrator approval...

Bot: Congrats! Your booking PK-EKSG9Z has been approved.
You:
```

That last line appears on its own, as soon as the administrator decides - whether through `admin_cli.py`, `/docs`, or a raw REST call - without interrupting whatever the user is typing. See [Notifying the user automatically](#notifying-the-user-automatically) under Stage 2 for how it works.

### Checking a booking's status

```
You: is PK-GC6ZMZ approved?
Bot: Your booking PK-GC6ZMZ is approved. The booking stands - each driver should show their pass ID on arrival.
```

If the user doesn't repeat the ID in a follow-up ("and now?"), the bot reuses the last booking ID mentioned in the conversation. If none is known yet, it asks for one rather than guessing.

### Cancelling is out of scope, by design

Submitted bookings cannot be cancelled or changed through the chatbot. Asked to, it answers immediately - no confirmation step - states the booking's current status, and gives the facility's real contact details for anything further:

```
You: please cancel booking PK-GC6ZMZ
Bot: Submitted bookings like PK-GC6ZMZ cannot be cancelled or changed through this
     chatbot. Your booking PK-GC6ZMZ is currently approved and remains exactly as it is.

     To cancel or change your reservation, please contact the parking operator's
     customer support:
     Phone: +1 555 010 2030
     Email: support@parkingfacility.example.com
```

### Guardrails in action

Prompt injection and attempts to obtain other customers' data are refused, and sensitive data is never revealed.

![Guardrails on input and output](screenshots/stage1/14_prompt_injection.png)

---

## Stage 2: human-in-the-loop approval

Every submitted booking is `pending` until an administrator decides. The chatbot and the administrator never call each other directly - they communicate only through the database and a typed message contract (`admin/models.py`: `ApprovalRequest`, `AdminDecision`), so either side can be swapped or tested independently.

### How a booking gets escalated

`submit_reservation` builds an `ApprovalRequest` and calls `AdminChannel.send_request()`. The real channel (`DbAdminChannel`) just marks the booking `escalated_at` in SQLite - a local write, not a network call - so submitting a reservation never fails because the admin server happens to be down. The admin server reads escalated, still-pending bookings whenever it's polled.

### Reviewing bookings: two ways to reach the same decision logic

**1. The admin console** (`scripts/admin_cli.py`) - a direct chat with the admin agent, no server required:

```bash
uv run python scripts/admin_cli.py
```

```
CityPark Central - Admin console. Type 'q' to quit.

Admin: what's pending?
Bot: 1 pending: PK-GC6ZMZ - Akshay Kumar ...

Admin: approve PK-GC6ZMZ, looks fine
Bot: Booking PK-GC6ZMZ - Akshay Kumar
     Period: 2026-12-31 09:00 to 2026-12-31 15:00
     Vehicles (1):
       - BV23NM3344 (car) - pass PK-GC6ZMZ-01
     Reply with APPROVE or REFUSE (optionally followed by a reason).

     You're about to APPROVE this booking (reason: looks fine). Reply yes/confirm to proceed, or no/cancel to back out.

Admin: yes
Bot: Done - booking PK-GC6ZMZ is now APPROVED.
```

Naming a booking with a clear decision in one message (as above) skips straight to the confirmation step; asking first ("what's pending?", "tell me about PK-GC6ZMZ") goes through the agent's `list_pending_requests`/`get_request_details` tools instead. Either way, nothing is written until an explicit yes/confirm.

**2. Swagger UI or any HTTP client**, via the admin REST server (`uv run python scripts/run_admin_server.py`, on `http://127.0.0.1:8001`) - for scripting, automation, or testing from the browser. Click **Authorize** on `/docs` and paste the bearer token, or:

```bash
curl -H "Authorization: Bearer $ADMIN_API_TOKEN" http://127.0.0.1:8001/requests
curl -X POST -H "Authorization: Bearer $ADMIN_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"reply": "approve"}' \
     http://127.0.0.1:8001/requests/PK-GC6ZMZ/decision
```

Both paths end up at the exact same `admin/agent.py::apply_decision()` - there is one place in the whole project a decision is ever written to the database.

### Interpreting the administrator's reply

Deciding on a booking is deliberately **not** left to the LLM - it is deterministic code, run on the administrator's raw message, that only ever produces an `approved`/`refused` write it can account for:

- **A message naming a booking ID with a clear decision** ("approve PK-X", "refuse PK-X, no space") creates a *proposal* (booking ID + decision + reason) and shows the full details, asking for a plain yes/confirm or no/cancel - nothing is written yet.
- **A bare yes/confirm**, with a proposal pending, writes exactly that stored proposal - not a fresh reading of the word "yes" itself, since "yes" alone always reads as an approval and would wrongly override a stored *refusal* waiting for confirmation.
- **A bare no/cancel**, with a proposal pending, clears it; nothing is written.
- **A message naming another booking ID (or a different decision)** replaces whatever proposal was pending.
- **Anything ambiguous or negated** ("PK-X, not sure", "don't approve PK-X", or a reply to a pending proposal that's neither a clear yes nor a clear no) is left exactly as-is and the administrator is asked to state it plainly - it is never treated as an approval.
- **A message with no booking ID and no pending proposal** (browsing, chit-chat) falls through to the admin agent, whose tools (`list_pending_requests`, `get_request_details`) only read data; its one write tool, `record_decision`, is a defense-in-depth fallback, not the normal path.

The REST API's free-text `reply` field goes through the same underlying keyword check (`admin/agent.py::interpret_reply`): unambiguous keywords (`approve`, `refuse`, `yes`, `no`, ...) are read deterministically, with no LLM call and no room for model error on the common case. An LLM is used only for genuinely unclear text reaching that endpoint, and even then its output schema makes `approved` an impossible value to produce - **ambiguity can only resolve to refused, or to asking for a clearer reply, never to a false approval.** "Do not approve" and similar negated phrasing are also caught before the plain keyword check, so they are never misread as a plain approval.

**On trusting the administrator's actual words, not an LLM's claim about them:** the agent's `record_decision` tool has no `admin_message` argument in its schema - the LLM cannot supply or forge the text a decision is checked against. The tool reads the administrator's real latest input from a plain per-conversation store that only the CLI's own input loop writes to, independently re-derives the decision from it, and refuses to write if that doesn't match what the tool call claims.

### Notifying the user automatically

Checking status by asking (`check_booking_status`) is one path; the other is automatic. `chatbot/cli.py` starts a daemon thread per session that polls `reservation_tools.pending_decisions()` every `ADMIN_NOTIFY_POLL_SECONDS` for bookings *this conversation* submitted, and prints a message - "Congrats! Your booking ... has been approved.", a refusal with its reason, or an expiry notice - the moment a booking it is tracking leaves `pending`. Each decided booking is announced once, then stops being tracked. This is pure polling against the shared database, not a push channel, so it has the same two-process independence as the rest of Stage 2: the notifier never calls the admin server, only reads the same SQLite database it already migrates and writes to.

### Security and reliability

- **Auth:** every endpoint except `/health` requires `Authorization: Bearer <ADMIN_API_TOKEN>`, checked in constant time (`secrets.compare_digest`). If the token isn't configured, every request is rejected - fail closed, never fail open. Swagger's **Authorize** button works because this is wired as a proper `HTTPBearer` security scheme, not a hand-read header.
- **No double processing:** deciding on a booking is one atomic database transaction; two simultaneous decisions on the same booking cannot both win (tested with real concurrent threads, not just asserted).
- **Idempotent retries:** `apply_decision()` treats a repeated call with the *same* decision for an already-decided booking as success, not a conflict - so the admin agent (or a retried REST call) calling it twice has no extra effect. A genuinely conflicting second decision still raises.
- **Concurrent access:** the chatbot, the admin console and the admin REST server can all have `data/parking.db` open at the same time. `db/database.py::get_connection()` sets WAL mode (so reads never block on a writer) and an explicit `busy_timeout` (so a write that does collide with another waits and retries instead of raising "database is locked"); see `tests/db/test_database.py` and the end-to-end concurrency check alongside `tests/db/test_admin.py::test_two_simultaneous_decisions_on_the_same_booking_cannot_both_win`.
- **Expiry:** a booking left `pending` past `ADMIN_REQUEST_EXPIRY_MINUTES` is lazily flipped to `expired` whenever it's next read - no background scheduler needed.
- **Schema migration on startup:** the admin server runs the same safe, additive-only migration every other entry point uses, so starting it against an older database adds the missing columns instead of crashing with "no such column."
- **Privacy:** the only tool that can look up a booking by ID (`check_booking_status`) returns nothing beyond that booking's bare status and decision reason - never a name, plate, or other detail, even for a booking ID that isn't "yours."

---

## Design details

### RAG pipeline

| Stage | Module |
|---|---|
| Ingestion | `loader.py` (reads markdown) and `scripts/ingest.py` (deterministic chunk ids, so re-running updates instead of duplicating) |
| Chunking | `chunker.py` (`RecursiveCharacterTextSplitter`, 300 characters, 50 overlap) |
| Embedding | `embeddings.py` |
| Retrieval | `rag/vector_store.py` (cached to one client - see note below), `rag/chain.py::retrieve` (top-K = 5) |
| Prompt construction | `rag/prompts.py` |
| Generation | `llm.py`, `rag/chain.py::generate_answer` |

The LLM, embedding and vector-store factories return the LangChain abstractions (`BaseChatModel`, `Embeddings`, `VectorStore`), so providers can be swapped in one place without touching callers. `get_vector_store()` is cached to a single instance: the agent can call `search_parking_info` more than once in one turn (one per topic in a multi-part question), and those calls run on separate threads - building a fresh Chroma client per call raced inside Chroma's own startup check and intermittently raised `AttributeError`/`ValueError`. Reusing one client removed the race.

### Reservation dialogue

A booking is for one vehicle or a **group** of up to `MAX_VEHICLES_PER_BOOKING` vehicles. One person books: first name, last name, start and end time are shared by every vehicle; each vehicle adds only its car number and vehicle type. Every value goes through `ReservationDraft` / `Vehicle` (`reservation/models.py`):

- **Names:** 2 to 50 letters, spaces, hyphens, apostrophes; normalised to title case.
- **Car number:** 4 to 10 letters or digits, spaces and hyphens ignored, upper-cased; each plate only once per booking.
- **Vehicle type:** one of `motorcycle`, `three_wheeler`, `car`, `van`. The agent maps words like "scooter", "auto-rickshaw", "SUV" or "minibus" onto these, and skips the question if the user already named the vehicle.
- **Period:** start not in the past and at most `MAX_ADVANCE_DAYS` ahead (default 90), end after start, at most 30 days long.

Invalid values are rejected with a reason and the agent asks again. The user can correct any field, add or remove vehicles, and discard the **in-progress draft** at any point (`cancel_reservation`) - which only ever affects a not-yet-submitted draft; it reports honestly ("nothing to discard") rather than claiming something happened if there is none, and it can never cancel an already-submitted booking (see [Stage 2](#stage-2-human-in-the-loop-approval)). The summary and an explicit "yes" are enforced in code, not only in the prompt.

**Capacity.** A booking is accepted only if, for its period, total capacity minus the vehicles of overlapping `pending` and `approved` bookings is at least the group size. The check runs at the summary and again at submission, inside the same database transaction as the insert, so two simultaneous bookings cannot both take the last spaces.

**IDs.** On submission the booking gets a random ID (`PK-7K3Q9X`) and each vehicle a pass ID derived from it (`PK-7K3Q9X-01`, `-02`, ...) for the driver to show on arrival. IDs use Python's `secrets` module, an alphabet without look-alikes (no 0/O, 1/I/L), and are unique in the database. The booking is stored in `reservations` (one row, status `pending` → `approved`/`refused`/`expired`) and its vehicles in `reservation_vehicles` (one row per pass).

---

## Guardrails

| Direction | Check | How |
|---|---|---|
| Input | Prompt injection, requests for the system prompt | Pattern rules |
| Input | Fishing for other customers' reservations or data | Pattern rules |
| Input | Sensitive data typed by the user (email, phone, card, IBAN, SSN) | Presidio pretrained NER model |
| Input | Oversized messages | Length limit |
| Output | Sensitive data in the reply | Presidio detection and masking (for example `<EMAIL_ADDRESS>`) |
| Output | System-prompt leakage | Marker check, reply replaced |

The facility's own public phone and email are allow-listed so they are not masked. The chatbot's reservation tools only work on the current conversation's draft and write new bookings - none of them can browse or search saved reservations. The one exception, `check_booking_status`, can look up any booking ID given to it, but by design returns nothing beyond that booking's status and decision reason - never a name, plate or other personal detail, so knowing an ID never exposes who it belongs to. A test proves this holds even when a second, unrelated conversation is given the exact ID. Contact details are never invented: the prompt requires calling `search_parking_info` to retrieve the real phone/email from the documents rather than asserting that "customer support" exists without saying how to reach it. Each guardrail is tested in isolation in `tests/guardrails/`.

---

## Testing

```bash
uv run pytest
```

All external services (LLM, embeddings, Chroma, the admin REST API) are replaced with fakes, and each test uses its own temporary SQLite database, so the suite runs offline and without any API key or a running admin server.

![Test results](screenshots/stage1/18_tests_passing.png)

---

## Evaluation

```bash
uv run python scripts/run_evaluation.py
```

This runs retrieval, generation and agent evaluation against your configured LLM and vector store, and writes `evaluation/REPORT.md` with the method, dataset corrections, Precision@K and Recall@K, latency (mean, p95, max) per stage, tool-routing accuracy, and honesty on unanswerable questions (asked 3 times because LLM answers vary). It takes a few minutes and needs a valid `.env` and network access to DIAL. The individual evaluations can also run on their own:

```bash
uv run python -m parking_chatbot.evaluation.retrieval_eval
uv run python -m parking_chatbot.evaluation.generation_eval
uv run python -m parking_chatbot.evaluation.agent_eval
```

![Evaluation run](screenshots/stage1/16_evaluation_run.png)

### Results summary

Measured on 30 hand-labelled retrieval questions, 18 agent questions and 7 unanswerable questions (Stage 1 scope; Stage 2's admin flow is covered by its own unit and live-conversation tests, not this evaluation). Results depend on the models you configure, so re-run the evaluation after changing them.

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | 0.67 | 0.63 |
| 3 | 0.59 | 0.97 |
| 5 | 0.45 | 1.00 |

| Stage | Mean | p95 | Max |
|---|---|---|---|
| Retrieval only | 1.11 s | 1.24 s | 1.41 s |
| Generation only (RAG chain) | 2.20 s | 2.82 s | 3.19 s |
| Full agent turn (including tool calls) | 3.47 s | 6.01 s | 7.99 s |

Tool routing was correct for 18 of 18 questions, and the bot answered honestly on all 7 unanswerable questions in each of 3 runs. The weakest point is Precision@1 (0.67): the top result comes from the wrong document about a third of the time. The full report, including metric definitions, dataset corrections and limitations, is in [`evaluation/REPORT.md`](evaluation/REPORT.md).

![Evaluation report tables](screenshots/stage1/17_evaluation_report.png)

Note on metrics: Precision@K and Recall@K are computed at different levels. Precision@K counts retrieved chunks (relevant chunks among the K retrieved, divided by K), while Recall@K counts source documents (expected documents found among the K retrieved chunks, divided by the number of expected documents). The two are therefore not directly comparable, and the values are best read as separate indicators of ranking purity and coverage.

---

## Project structure

```
parking_chatbot/
  config.py              settings from environment variables
  llm.py / embeddings.py  DIAL chat and embedding model factories
  loader.py / chunker.py  static-doc ingestion pipeline
  rag/                    vector store, retrieval + generation chain, prompts
  db/
    database.py           connection + transaction helper
    schema.py              table definitions and migrations (safe to run repeatedly)
    queries.py             working hours, prices, availability
    reservations.py        booking creation, capacity check
    admin.py                pending bookings, status lookup, recording a decision
  reservation/            ReservationDraft/Vehicle domain model, booking/pass ID generation
  admin/                  Stage 2: the human-in-the-loop channel
    models.py              ApprovalRequest / AdminDecision - the message contract
    channel.py              AdminChannel protocol, the real (DB) and fake (test) implementations
    agent.py                 the admin agent: tools, system prompt, the deterministic
                               confirmation flow, and the one shared write path
    server.py                 the REST API (FastAPI), reusing that same write path
  guardrails/              input guard, output guard, Presidio PII helpers
  chatbot/                 chat agent, tools (including check_booking_status), CLI
  evaluation/              dataset, metrics, retrieval/generation/agent eval, report
scripts/
  seed_db.py               create tables + demo data
  ingest.py                build the vector store
  run_admin_server.py      start the admin REST API
  admin_cli.py             conversational console for the administrator (chats with
                             the admin agent directly - no admin server needed)
  run_evaluation.py        run all evaluations, write evaluation/REPORT.md
  inspect_data.py           print SQL tables + a Chroma sample, for inspection
data/static/              source documents for the vector store
tests/                    pytest suite (offline, external services faked)
stages/                   stage-by-stage requirements
```

---

## Known limitations

- Conversation history and reservation drafts are held in memory, so they last for one CLI session. Persistent checkpointing is planned for Stage 4.
- The limits quoted in `data/static/*.md` (90 days, 10 vehicles) are plain text. If you change `MAX_ADVANCE_DAYS` or `MAX_VEHICLES_PER_BOOKING`, edit the docs too and re-run `ingest.py` (`tests/test_docs_match_config.py` fails until they agree).
- The chat model must support tool calling, and answer quality and latency vary with the model you choose.
- There is one shared `ADMIN_API_TOKEN` for all administrators - no per-admin accounts or audit trail of *who* decided, only *that* a decision was made and when.
- A plate can appear in two overlapping bookings without being flagged - refusing on that basis would reveal to a stranger that the plate is already booked, which the design deliberately avoids.
- Recording approved bookings to a file (Stage 3) and full LangGraph orchestration (Stage 4) have not started yet.
