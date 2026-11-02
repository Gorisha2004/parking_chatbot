"""Run the admin API locally: uv run python scripts/run_admin_server.py"""
import uvicorn


def main() -> None:
    uvicorn.run("parking_chatbot.admin.server:app", host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
