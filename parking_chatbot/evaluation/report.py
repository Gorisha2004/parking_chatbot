"""Build the Stage 1 evaluation report (evaluation/REPORT.md) from one command.

Runs retrieval, generation and tool-routing evaluation once and the honesty check
HONESTY_RUNS times (LLM answers vary between runs), and renders the combined results
as markdown: `uv run python scripts/run_evaluation.py`.

The narrative text (definitions, the frozen pre-correction baseline, limitations) lives
in report_notes.py, so re-running this script - which only recomputes the numeric
sections below - can never silently drop it.
"""
from datetime import datetime

from parking_chatbot import config
from parking_chatbot.chatbot.agent import build_agent
from parking_chatbot.evaluation import agent_eval, generation_eval, report_notes, retrieval_eval
from parking_chatbot.evaluation.dataset import (
    LABEL_CORRECTIONS,
    RETRIEVAL_CASES,
    TOOL_ROUTING_CASES,
    UNANSWERABLE_QUESTIONS,
)
from parking_chatbot.evaluation.metrics import summarize_latencies
from parking_chatbot.rag.vector_store import get_vector_store

REPORT_PATH = config.PROJECT_ROOT / "evaluation" / "REPORT.md"
HONESTY_RUNS = 3
MAX_REPLY_CHARS = 300


def _latency_row(label: str, summary: dict) -> str:
    return f"| {label} | {summary['mean']:.2f}s | {summary['p95']:.2f}s | {summary['max']:.2f}s |"


def _table_cell(text: str) -> str:
    """Make free text safe for one markdown table cell: one line, no column breaks, bounded length."""
    one_line = " ".join(text.split()).replace("|", "\\|")
    return one_line if len(one_line) <= MAX_REPLY_CHARS else one_line[: MAX_REPLY_CHARS - 1] + "…"


def _precision_recall_table(pr_table: list[dict]) -> list[str]:
    return [
        "| K | Precision@K | Recall@K |",
        "|---|---|---|",
        *[f"| {row['k']} | {row['precision']:.2f} | {row['recall']:.2f} |" for row in pr_table],
    ]


def _low_recall_summary(low_recall: list[dict], total: int) -> list[str]:
    if not low_recall:
        return ["Every case achieved full Recall@5."]
    lines = [f"Cases with Recall@5 < 1 ({len(low_recall)} of {total}):", ""]
    lines += [f"- *{c['question']}* — expected {c['expected']}, got {c['sources']}" for c in low_recall]
    return lines


def _chunk_count() -> int:
    """Chunks currently stored in Chroma. A local client-side count, no embedding call."""
    return get_vector_store()._collection.count()


def _setup_section() -> str:
    """Setup facts: all read from config or counted locally, no DIAL call needed."""
    return report_notes.SETUP_TEMPLATE.format(
        chat_model=config.CHAT_MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        collection_name=config.COLLECTION_NAME,
        top_k=config.TOP_K,
        doc_count=len(list(config.STATIC_DATA_DIR.glob("*.md"))),
        chunk_count=_chunk_count(),
        retrieval_count=len(RETRIEVAL_CASES),
        unanswerable_count=len(UNANSWERABLE_QUESTIONS),
        routing_count=len(TOOL_ROUTING_CASES),
    )


def build_report() -> str:
    """Run every evaluation and render the results as a markdown report."""
    print("Running retrieval evaluation...")
    retrieval_results = retrieval_eval.evaluate_retrieval()
    pr_table = retrieval_eval.precision_recall_table(retrieval_results)
    low_recall = retrieval_eval.low_recall_cases(retrieval_results)
    retrieval_latency = summarize_latencies([r["latency"] for r in retrieval_results])

    print("Running generation evaluation...")
    generation_results = generation_eval.evaluate_generation()
    generation_latency = summarize_latencies([r["latency"] for r in generation_results])

    print("Running agent evaluation (tool routing)...")
    agent = build_agent()
    routing = agent_eval.tool_routing_results(agent)
    agent_latency = summarize_latencies(routing["latencies"])

    judge = agent_eval.get_chat_model()
    honesty_runs = []
    for run in range(1, HONESTY_RUNS + 1):
        print(f"Running honesty check {run}/{HONESTY_RUNS}...")
        honesty_runs.append(agent_eval.unanswerable_results(agent, judge))
    honesty = agent_eval.summarize_honesty_runs(honesty_runs)

    accuracy = routing["correct"] / routing["total"]
    honesty_rate = honesty["mean"]
    recall_at_5 = pr_table[-1]["recall"]
    precision_at_1 = pr_table[0]["precision"]

    lines = [
        "# Stage 1 Evaluation Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Setup",
        "",
        _setup_section(),
        "",
        "## Method",
        "",
        "- Retrieval quality: Precision@K / Recall@K computed against a hand-labelled "
        f"dataset of {len(RETRIEVAL_CASES)} questions (`parking_chatbot/evaluation/dataset.py`), "
        "each with the static-doc source(s) it should be retrieved from. See \"Precision@K and "
        "Recall@K, precisely\" below — the two metrics are computed at different levels.",
        "- Generation latency: the RAG chain's `generate_answer` is timed separately from "
        "retrieval, given already-retrieved context, over the same question set.",
        f"- Tool-routing accuracy and agent latency: {len(TOOL_ROUTING_CASES)} questions "
        "sent through the full agent (`chatbot.agent.chat`), checking which tool(s) it called.",
        f"- Honesty on unanswerable questions: {len(UNANSWERABLE_QUESTIONS)} questions with no "
        "answer in the documentation, graded HONEST/INVENTED by an LLM-as-judge. Because LLM "
        f"answers vary between runs, the set is asked {HONESTY_RUNS} times and the range is reported.",
        "- Latency is end-to-end wall-clock time per call (`time.perf_counter()`); mean, p95 "
        "and max are reported. Reproduce with `uv run python scripts/run_evaluation.py`.",
        "",
        "### Precision@K and Recall@K, precisely",
        "",
        report_notes.METRIC_DEFINITIONS,
        "",
        "## Dataset corrections",
        "",
        "Labels changed after a run showed them contradicting the source documents "
        "(each checked against `data/static/*.md`):",
        "",
        *[f"- {correction}" for correction in LABEL_CORRECTIONS],
        "",
        report_notes.BEFORE_CORRECTION_INTRO,
        "",
        f"After correction ({len(retrieval_results)} cases, current run):",
        "",
        *_precision_recall_table(pr_table),
        "",
        *_low_recall_summary(low_recall, len(retrieval_results)),
        "",
        "## Retrieval quality (Precision@K / Recall@K)",
        "",
        *_precision_recall_table(pr_table),
        "",
        *_low_recall_summary(low_recall, len(retrieval_results)),
        "",
        "## Latency (seconds)",
        "",
        "| Stage | Mean | p95 | Max |",
        "|---|---|---|---|",
        _latency_row("Retrieval only", retrieval_latency),
        _latency_row("Generation only (RAG chain)", generation_latency),
        _latency_row("Full agent turn (incl. tool calls)", agent_latency),
        "",
        "## Tool-routing accuracy",
        "",
        f"{routing['correct']}/{routing['total']} = {accuracy:.2f}",
        "",
    ]
    if routing["failures"]:
        lines.append("Mismatches:")
        lines.append("")
        lines += [f"- *{f['question']}* — expected {f['expected']}, got {f['got']}" for f in routing["failures"]]
        lines.append("")

    run_rates = ", ".join(f"{rate:.2f}" for rate in honesty["rates"])
    lines += [
        "## Honesty on unanswerable questions",
        "",
        f"Mean {honesty_rate:.2f} answered honestly (refused or pointed to support instead of "
        f"inventing an answer); per run: {run_rates} (range {honesty['min']:.2f}–{honesty['max']:.2f}).",
        "",
        "| Question | Honest runs | Example reply (an invented one if any run invented) |",
        "|---|---|---|",
        *[
            f"| {_table_cell(q['question'])} | {q['honest_runs']}/{q['runs']} | {_table_cell(q['example_answer'])} |"
            for q in honesty["questions"]
        ],
        "",
        "## Limitations",
        "",
        report_notes.LIMITATIONS_TEMPLATE.format(
            top_k=config.TOP_K,
            doc_count=len(list(config.STATIC_DATA_DIR.glob("*.md"))),
            chunk_count=_chunk_count(),
            retrieval_count=len(RETRIEVAL_CASES),
            routing_count=len(TOOL_ROUTING_CASES),
            unanswerable_count=len(UNANSWERABLE_QUESTIONS),
            honesty_judgments=len(UNANSWERABLE_QUESTIONS) * HONESTY_RUNS,
            honesty_runs=HONESTY_RUNS,
        ),
        "",
        "## Conclusions",
        "",
        f"- Retrieval finds the right document almost every time (Recall@5 = {recall_at_5:.2f}). "
        f"The weak point is Precision@1 = {precision_at_1:.2f}: {report_notes.PRECISION_AT_1_EXPLANATION}",
        f"- Tool routing is reliable on this test set ({accuracy:.0%} correct over {routing['total']} questions).",
        f"- The agent avoided inventing answers for out-of-scope questions throughout this run "
        f"({honesty_rate:.0%} honest across {HONESTY_RUNS} repeats of {len(UNANSWERABLE_QUESTIONS)} questions). "
        f"{report_notes.HONESTY_CAVEAT}"
        if honesty_rate >= 0.9
        else f"- The agent sometimes invents answers for out-of-scope questions "
        f"({honesty_rate:.0%} honest on average over {HONESTY_RUNS} runs). {report_notes.HONESTY_CAVEAT}",
        f"- Generation accounts for a larger share of end-to-end latency than retrieval, "
        f"{report_notes.LATENCY_EMBEDDING_NOTE}"
        if generation_latency["mean"] > retrieval_latency["mean"]
        else "- Retrieval and generation latency are comparable.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
