import statistics
from pathlib import Path
import time

from parking_chatbot.evaluation.dataset import RETRIEVAL_CASES
from parking_chatbot.evaluation.metrics import summarize_latencies
from parking_chatbot.rag.chain import retrieve

K_VALUES = (1, 3, 5)

def precision_at_k(retrieved: list[str], expected: list[str]) -> float:
    """Share of the retrieved chunks that come from an expected source."""
    if not retrieved:
        return 0.0
    return sum(source in expected for source in retrieved) / len(retrieved)


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    """Share of the expected sources that appear among the retrieved chunks."""
    return len(set(retrieved) & set(expected)) / len(set(expected))

def evaluate_retrieval() -> list[dict]:
    """Run every case once at the largest K and time the search."""
    results = []
    for case in RETRIEVAL_CASES:
        start = time.perf_counter()
        docs = retrieve(case["question"], k=max(K_VALUES))
        latency = time.perf_counter() - start
        sources = [Path(doc.metadata["source"]).stem for doc in docs]
        results.append({**case, "sources": sources, "latency": latency})
    return results


def precision_recall_table(results: list[dict]) -> list[dict]:
    """Per-K Precision@K / Recall@K averaged across all cases."""
    table = []
    for k in K_VALUES:
        precision = statistics.mean(precision_at_k(r["sources"][:k], r["expected"]) for r in results)
        recall = statistics.mean(recall_at_k(r["sources"][:k], r["expected"]) for r in results)
        table.append({"k": k, "precision": precision, "recall": recall})
    return table


def low_recall_cases(results: list[dict]) -> list[dict]:
    """Cases where Recall@5 < 1, i.e. at least one expected source was missed."""
    return [r for r in results if recall_at_k(r["sources"], r["expected"]) < 1]


def print_report(results: list[dict]) -> None:
    print(f"Questions evaluated: {len(results)}\n")
    print(" K | Precision@K | Recall@K")
    for row in precision_recall_table(results):
        print(f"{row['k']:>2} | {row['precision']:11.2f} | {row['recall']:8.2f}")

    summary = summarize_latencies([r["latency"] for r in results])
    print(f"\nRetrieval latency: mean {summary['mean']:.2f}s, p95 {summary['p95']:.2f}s, max {summary['max']:.2f}s")

    print("\nCases where Recall@5 < 1:")
    for r in low_recall_cases(results):
        print(f"- {r['question']}\n  expected {r['expected']}, got {r['sources']}")


if __name__ == "__main__":
    print_report(evaluate_retrieval())
