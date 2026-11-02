"""Evaluation of the RAG chain's generation stage, timed separately from retrieval."""
import time

from parking_chatbot.evaluation.dataset import RETRIEVAL_CASES
from parking_chatbot.evaluation.metrics import summarize_latencies
from parking_chatbot.rag.chain import generate_answer, retrieve


def evaluate_generation() -> list[dict]:
    """Retrieve once per question (untimed), then time pure LLM generation."""
    results = []
    for case in RETRIEVAL_CASES:
        docs = retrieve(case["question"])
        start = time.perf_counter()
        answer = generate_answer(case["question"], docs)
        latency = time.perf_counter() - start
        results.append({"question": case["question"], "answer": answer, "latency": latency})
    return results


def print_report(results: list[dict]) -> None:
    summary = summarize_latencies([r["latency"] for r in results])
    print(f"Questions evaluated: {len(results)}")
    print(f"Generation latency: mean {summary['mean']:.2f}s, p95 {summary['p95']:.2f}s, max {summary['max']:.2f}s")


if __name__ == "__main__":
    print_report(evaluate_generation())
