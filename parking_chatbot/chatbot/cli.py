import argparse
import threading
import time
import traceback
import uuid

from parking_chatbot import config
from parking_chatbot.chatbot.agent import build_agent, chat
from parking_chatbot.chatbot.reservation_tools import pending_decisions

EXIT_COMMANDS = {"exit", "quit"}


def _decision_message(booking: dict) -> str:
    booking_id = booking["booking_id"]
    if booking["status"] == "approved":
        return f"Congrats! Your booking {booking_id} has been approved."
    if booking["status"] == "refused":
        reason = f" Reason: {booking['decision_reason']}" if booking["decision_reason"] else ""
        return f"Your booking {booking_id} was refused.{reason}"
    return f"Your booking {booking_id} expired before a decision was made."


def _announce_decisions(thread_id: str) -> None:
    for booking in pending_decisions(thread_id):
        print(f"\nBot: {_decision_message(booking)}\nYou: ", end="", flush=True)


def _watch_for_decisions(thread_id: str, stop_event: threading.Event) -> None:
    """Runs on a background thread for the life of the session, so the user is told
    about an administrator's decision as soon as it happens, without having to ask."""
    while not stop_event.wait(config.ADMIN_NOTIFY_POLL_SECONDS):
        _announce_decisions(thread_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CityPark Central assistant")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the tools used, response time and full error details",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_agent()
    thread_id = str(uuid.uuid4())  # one new conversation per program run
    print("CityPark Central assistant. Type 'exit' to quit.\n")

    stop_notifier = threading.Event()
    notifier = threading.Thread(target=_watch_for_decisions, args=(thread_id, stop_notifier), daemon=True)
    notifier.start()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in EXIT_COMMANDS:
                print("Bye!")
                break

            start = time.perf_counter()
            try:
                answer, tools = chat(agent, user_input, thread_id)
            except Exception as exc:
                print(f"Bot: Sorry, something went wrong ({type(exc).__name__}). Please try again.\n")
                if args.debug:
                    traceback.print_exc()
                continue
            elapsed = time.perf_counter() - start

            print(f"Bot: {answer}")
            if args.debug:
                print(f"     [tools: {tools or 'none'} | {elapsed:.2f}s]")
            print()
    finally:
        stop_notifier.set()


if __name__ == "__main__":
    main()