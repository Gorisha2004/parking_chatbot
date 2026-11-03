"""The single command that runs the whole system.

One process, one LangGraph workflow, one conversation: the user asks questions,
makes a reservation, hears the administrator's decision when it arrives, and -
if it was approved - sees it confirmed as recorded.

The administrator is a person working elsewhere (the admin console or the admin
REST API, Stage 2), and the recording server is a separate process (Stage 3), so
this CLI never waits on either. It drives the graph on two threads instead:

* the input loop, for everything the user types;
* a background ticker, which runs a turn with no user input whenever a booking is
  pending, so a decision reaches the user without them having to ask.

Both go through the same graph and the same checkpointed thread, so they are
serialized by one lock - a checkpointed state must only ever have one writer.
"""
import argparse
import logging
import threading
import time
import traceback
import uuid
from typing import Any

from parking_chatbot import config
from parking_chatbot.db.schema import create_tables
from parking_chatbot.orchestration.checkpointing import build_checkpointer, close_checkpointer
from parking_chatbot.orchestration.graph import build_workflow, run_turn, workflow_state
from parking_chatbot.orchestration.state import WorkflowState, has_pending_booking, is_awaiting_recording

logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit", "q"}
BANNER = "CityPark Central assistant. Type 'exit' to quit."
TURN_FAILED = "Sorry, something went wrong ({error}). Your reservation is unaffected - please try again."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CityPark Central - the complete parking assistant")
    parser.add_argument("--debug", action="store_true", help="show the tools used, response time and full error details")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="resume an earlier conversation by its id (needs the sqlite checkpoint backend)",
    )
    parser.add_argument(
        "--checkpoint-backend",
        choices=("sqlite", "memory"),
        default=None,
        help="override WORKFLOW_CHECKPOINT_BACKEND for this run",
    )
    return parser.parse_args(argv)


def render(state: WorkflowState) -> list[str]:
    """Everything this turn has to say to the user, in order.

    The agent's own answer comes first, then any notice the workflow itself
    produced (submitted, approved, refused, expired, recording failed,
    cancelled) - so an unhappy path always ends in an explicit message and never
    in silence.
    """
    lines = []
    if state.get("response"):
        lines.append(state["response"])
    lines.extend(state.get("notices") or [])
    return lines


def _print_turn(lines: list[str]) -> None:
    for line in lines:
        print(f"Bot: {line}")
    if lines:
        print()


def _print_interrupting(lines: list[str]) -> None:
    """Print from the background ticker, then restore the prompt the user is typing at."""
    if not lines:
        return
    print()
    for line in lines:
        print(f"Bot: {line}")
    print("You: ", end="", flush=True)


class WorkflowSession:
    """One conversation: the graph, its thread id, and the lock that serializes it."""

    def __init__(self, workflow: Any, thread_id: str) -> None:
        self._workflow = workflow
        self._lock = threading.Lock()
        self.thread_id = thread_id

    def send(self, user_input: str) -> list[str]:
        """Run a turn for something the user typed."""
        with self._lock:
            return render(run_turn(self._workflow, self.thread_id, user_input))

    def tick(self) -> list[str]:
        """Run a turn with no user input, but only if something is actually waiting.

        Skipping otherwise is what keeps the ticker free: a tick on an idle
        conversation does no database read and never reaches the LLM.
        """
        with self._lock:
            state = workflow_state(self._workflow, self.thread_id)
            if not (has_pending_booking(state) or is_awaiting_recording(state)):
                return []
            return render(run_turn(self._workflow, self.thread_id, "", turn_kind="tick"))


def _watch_for_decisions(session: WorkflowSession, stop_event: threading.Event) -> None:
    """Poll for an administrator decision for the life of the session."""
    while not stop_event.wait(config.ADMIN_NOTIFY_POLL_SECONDS):
        try:
            _print_interrupting(session.tick())
        except Exception as error:  # a failed tick must never kill the session
            logger.warning("Background decision check failed (%s: %s)", type(error).__name__, error)


def run_loop(session: WorkflowSession, debug: bool = False) -> None:
    """Read, run, print - until the user leaves."""
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Bye!")
            return

        started = time.perf_counter()
        try:
            lines = session.send(user_input)
        except Exception as error:
            # The turn failed, so nothing it would have written was committed:
            # the checkpointed state is still the one from before this message.
            print(f"Bot: {TURN_FAILED.format(error=type(error).__name__)}\n")
            logger.exception("Turn failed for thread %s", session.thread_id)
            if debug:
                traceback.print_exc()
            continue

        _print_turn(lines)
        if debug:
            print(f"     [{time.perf_counter() - started:.2f}s]\n")


def main(argv: list[str] | None = None) -> None:
    """Build the whole system and run it until the user exits."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    create_tables()  # safe every time; only adds missing tables and columns

    checkpointer = build_checkpointer(backend=args.checkpoint_backend)
    try:
        workflow = build_workflow(checkpointer=checkpointer)
        thread_id = args.thread_id or str(uuid.uuid4())
        session = WorkflowSession(workflow, thread_id)

        print(BANNER)
        if args.debug:
            print(f"     [conversation id: {thread_id}]")
        print()

        stop_ticker = threading.Event()
        ticker = threading.Thread(target=_watch_for_decisions, args=(session, stop_ticker), daemon=True)
        ticker.start()
        try:
            run_loop(session, debug=args.debug)
        finally:
            stop_ticker.set()
    finally:
        close_checkpointer(checkpointer)


if __name__ == "__main__":
    main()
