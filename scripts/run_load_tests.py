"""Stage 4 load tests: one component at a time, at a few concurrency levels,
reported as markdown.

    uv run python scripts/run_load_tests.py              # fake LLM (default, reproducible, no network)
    uv run python scripts/run_load_tests.py --live        # chatbot scenario uses the real DIAL-backed agent

Everything runs against a disposable SQLite database and a disposable
reservation file under a temp directory - never against data/parking.db or
data/reservations.txt - so this is always safe to run, repeatedly, on a
developer machine or in CI.
"""
import argparse
import tempfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

from parking_chatbot import config
from parking_chatbot.db.database import get_connection
from parking_chatbot.db.schema import create_tables
from parking_chatbot.evaluation.load import summarize
from parking_chatbot.evaluation.load_scenarios import (
    admin_confirmation_scenario,
    chatbot_dialogue_scenario,
    mcp_recording_scenario,
)

REPORT_PATH = config.PROJECT_ROOT / "docs" / "LOAD_TEST_REPORT.md"
#: (sessions_or_calls, concurrency) pairs tried for every component, light to heavy.
LOAD_LEVELS = [(10, 1), (40, 5), (120, 20)]


def _seed_capacity(total_spaces: int) -> None:
    with closing(get_connection()) as conn, conn:
        conn.execute("DELETE FROM availability")
        conn.execute(
            "INSERT INTO availability (level, total_spaces, available_spaces) VALUES (1, ?, ?)",
            (total_spaces, total_spaces),
        )


def _row(component: str, level: tuple[int, int], stats: dict) -> str:
    return (
        f"| {component} | {level[0]} | {level[1]} | {stats['mean']:.3f}s | {stats['p50']:.3f}s | "
        f"{stats['p95']:.3f}s | {stats['p99']:.3f}s | {stats['max']:.3f}s | "
        f"{stats['error_rate']:.1%} | {stats['throughput_per_second']:.1f}/s |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4 component load tests")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the real DIAL-backed chat agent for the chatbot scenario instead of a fake one (needs VPN/credentials)",
    )
    return parser.parse_args()


def run_all(use_live_llm: bool) -> list[str]:
    """Run every component at every load level; return the report's table rows."""
    rows = []
    for calls, concurrency in LOAD_LEVELS:
        result = chatbot_dialogue_scenario(sessions=calls, concurrency=concurrency, use_live_llm=use_live_llm)
        rows.append(_row("Chatbot (dialogue, 3 turns/session)", (calls, concurrency), summarize(result)))

    # Seeded once, big enough for every level's bookings combined: approved and
    # pending bookings all stay "active" (and so keep counting against
    # capacity) for the rest of the run, since nothing here ever cancels one.
    _seed_capacity(total_spaces=sum(calls for calls, _ in LOAD_LEVELS) + 20)
    for calls, concurrency in LOAD_LEVELS:
        result = admin_confirmation_scenario(decisions=calls, concurrency=concurrency)
        rows.append(_row("Administrator confirmation", (calls, concurrency), summarize(result)))

    for calls, concurrency in LOAD_LEVELS:
        result, integrity = mcp_recording_scenario(bookings=calls, concurrency=concurrency)
        stats = summarize(result)
        note = "OK" if integrity["matches"] else f"MISMATCH ({integrity['lines_written']}/{integrity['expected_lines']} lines)"
        rows.append(_row(f"MCP recording [{note}]", (calls, concurrency), stats))
    return rows


def write_report(rows: list[str], use_live_llm: bool) -> None:
    llm_mode = "the real DIAL-backed agent (--live)" if use_live_llm else "a fake agent (no model call; the default)"
    lines = [
        "# Stage 4 Load Test Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Method",
        "",
        "- Each component is driven directly through the same function the orchestration "
        "nodes call in production (`parking_chatbot.evaluation.load_scenarios`), not through "
        "a network boundary, except the MCP recording scenario, which still goes through the "
        "real MCP tool, called in-process the same way `tests/recording/test_recording_server.py` "
        "does - so what is measured is real component logic, not a mock's overhead.",
        f"- **Chatbot**: the LLM is {llm_mode}. The fake agent returns a fixed reply with no "
        "model call, so its numbers measure `chatbot.agent.chat()`'s own overhead (the input "
        "guard, the greeting check, the output guard's PII scan) - not real model latency. "
        "Each \"call\" is one whole 3-turn session, run against a fresh thread id, mirroring "
        "several users each having a short conversation at once.",
        "- **Administrator confirmation**: `apply_decision()`, the one real write path Stage 2 "
        "ever uses - no LLM is involved in this path at all. Each call decides one real pending "
        "booking, pre-seeded with enough availability for the whole run.",
        "- **MCP recording**: the real `record_reservation` MCP tool, writing to a disposable "
        "reservation file guarded by its real file lock. Reported alongside each row is whether "
        "the file ended up with exactly the expected number of lines (OK) or not (MISMATCH).",
        "- All three run against a disposable SQLite database and reservation file created fresh "
        "for this run, never the project's real `data/parking.db` or `data/reservations.txt`.",
        "- Concurrency is simulated with a thread pool (`concurrent.futures.ThreadPoolExecutor`); "
        "every call's own latency is timed with `time.perf_counter()`, and a failing call is "
        "recorded as an error rather than aborting the run.",
        "- Reproduce with `uv run python scripts/run_load_tests.py` (add `--live` for the real "
        "chat agent).",
        "",
        "## Load levels",
        "",
        "Each component is run at three levels: light (10 calls, concurrency 1 - a baseline with "
        "no contention), moderate (40 calls, concurrency 5) and heavy (120 calls, concurrency 20).",
        "",
        "## Results",
        "",
        "| Component | Calls | Concurrency | Mean | p50 | p95 | p99 | Max | Error rate | Throughput |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## Findings",
        "",
        "- *(fill in after reading the numbers above for this run - latency trend as concurrency "
        "rises, whether any component's error rate is non-zero, and the MCP file-integrity check "
        "result.)*",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="parking-load-test-") as tmp_dir:
        config.SQL_DB_PATH = f"{tmp_dir}/load_test.db"
        config.MCP_RESERVATIONS_FILE_PATH = Path(tmp_dir) / "load_test_reservations.txt"
        create_tables()

        print("Running chatbot, administrator-confirmation and MCP-recording load scenarios...")
        rows = run_all(use_live_llm=args.live)

    write_report(rows, use_live_llm=args.live)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
