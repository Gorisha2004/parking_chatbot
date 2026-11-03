"""Start the complete system: the LangGraph workflow, end to end, in one command.

    uv run python scripts/run_workflow.py

A thin launcher, so the orchestration package holds no script-only code. See
parking_chatbot/orchestration/cli.py for what it starts, and the README for the
two companion processes (the admin console and the MCP recording server).
"""
from parking_chatbot.orchestration.cli import main

if __name__ == "__main__":
    main()
