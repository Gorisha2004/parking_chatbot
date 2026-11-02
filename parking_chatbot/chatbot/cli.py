import argparse
import time
import traceback
import uuid

from parking_chatbot.chatbot.agent import build_agent, chat

EXIT_COMMANDS = {"exit", "quit"}


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


if __name__ == "__main__":
    main()