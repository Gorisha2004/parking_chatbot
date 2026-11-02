"""Conversational console for the administrator: chat with the admin agent directly.

No admin server needs to be running - this talks to the same database the chatbot
and the admin REST server use, through the admin agent's own tools.

Run: uv run python scripts/admin_cli.py
"""
import argparse
import traceback
import uuid

from parking_chatbot.admin.agent import admin_chat, build_admin_agent

EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CityPark Central admin console")
    parser.add_argument("--debug", action="store_true", help="show the tools used and full error details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = build_admin_agent()
    thread_id = str(uuid.uuid4())  # one new conversation per program run
    print("CityPark Central - Admin console. Type 'q' to quit.\n")

    while True:
        try:
            admin_input = input("Admin: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not admin_input:
            continue
        if admin_input.lower() in EXIT_COMMANDS:
            print("Bye!")
            break

        try:
            answer, tools = admin_chat(agent, admin_input, thread_id)
        except Exception as exc:
            print(f"Bot: Sorry, something went wrong ({type(exc).__name__}). Please try again.\n")
            if args.debug:
                traceback.print_exc()
            continue

        print(f"Bot: {answer}")
        if args.debug:
            print(f"     [tools: {tools or 'none'}]")
        print()


if __name__ == "__main__":
    main()
