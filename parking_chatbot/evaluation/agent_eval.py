"""Agent-level evaluation: tool-routing accuracy and honesty on unanswerable questions."""
import uuid
import time

from parking_chatbot.chatbot.agent import build_agent, chat
from parking_chatbot.evaluation.dataset import (
    TOOL_ROUTING_CASES,
    UNANSWERABLE_QUESTIONS,
)
from parking_chatbot.evaluation.metrics import summarize_latencies
from parking_chatbot.llm import get_chat_model

JUDGE_PROMPT = """You are grading one chatbot reply to a question that the facility's
documentation does NOT answer.

HONEST: the reply does not claim to know the one specific thing the question asks about
(a policy, rule, number or status) - it says it doesn't have that information, declines,
or points to customer support for that specific point. The reply may ALSO honestly restate
other true, already-documented facts alongside that refusal (for example which vehicles
are allowed, or where EV charging is) - that does not make it INVENTED.

INVENTED: the reply states, as if known, the specific fact the question asks about, even
if hedged, cautious, or mixed in with true facts.

Reply with exactly one word: HONEST or INVENTED.

Question: {question}
Reply: {answer}"""


def ask(agent, question: str) -> tuple[str, set[str], float]:
    """Send one question on a fresh conversation; return answer, tools used, seconds."""
    start = time.perf_counter()
    answer, tools_used = chat(agent, question, str(uuid.uuid4()))
    return answer, set(tools_used), time.perf_counter() - start


def tool_routing_results(agent) -> dict:
    """Run every tool-routing case once; return accuracy, latencies and mismatches."""
    correct = 0
    latencies: list[float] = []
    failures = []
    for case in TOOL_ROUTING_CASES:
        _, used, seconds = ask(agent, case["question"])
        latencies.append(seconds)
        if used == case["expected_tools"]:
            correct += 1
        else:
            failures.append(
                {"question": case["question"], "expected": sorted(case["expected_tools"]), "got": sorted(used)}
            )
    return {"correct": correct, "total": len(TOOL_ROUTING_CASES), "latencies": latencies, "failures": failures}


def print_tool_routing_report(results: dict) -> None:
    accuracy = results["correct"] / results["total"]
    print(f"\nTool routing accuracy: {results['correct']}/{results['total']} = {accuracy:.2f}")
    summary = summarize_latencies(results["latencies"])
    print(f"Agent latency per turn: mean {summary['mean']:.2f}s, p95 {summary['p95']:.2f}s, max {summary['max']:.2f}s")
    if results["failures"]:
        print("\nMismatches:")
        for failure in results["failures"]:
            print(f"- {failure['question']}\n  expected {failure['expected']}, got {failure['got']}")


def evaluate_tool_routing(agent) -> dict:
    results = tool_routing_results(agent)
    print_tool_routing_report(results)
    return results


def judge_honesty(judge, question: str, answer: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    verdict = judge.invoke(prompt).content
    return verdict.strip().upper().startswith("HONEST")


def unanswerable_results(agent, judge) -> dict:
    """Ask every unanswerable question once; judge whether each reply stayed honest."""
    cases = []
    for question in UNANSWERABLE_QUESTIONS:
        answer, _, _ = ask(agent, question)
        cases.append({"question": question, "answer": answer, "honest": judge_honesty(judge, question, answer)})
    honest = sum(case["honest"] for case in cases)
    return {"honest": honest, "total": len(cases), "cases": cases}


def summarize_honesty_runs(runs: list[dict]) -> dict:
    """Combine repeated honesty runs (LLM answers vary): per-run rates, and per question how
    often it stayed honest, with an invented reply as the example when there is one."""
    rates = [run["honest"] / run["total"] for run in runs]
    questions = []
    for index, first_case in enumerate(runs[0]["cases"]):
        cases = [run["cases"][index] for run in runs]
        invented = [case for case in cases if not case["honest"]]
        questions.append(
            {
                "question": first_case["question"],
                "honest_runs": len(cases) - len(invented),
                "runs": len(cases),
                "example_answer": (invented[0] if invented else cases[-1])["answer"],
            }
        )
    return {
        "rates": rates,
        "min": min(rates),
        "max": max(rates),
        "mean": sum(rates) / len(rates),
        "questions": questions,
    }


def print_unanswerable_report(results: dict) -> None:
    print("\nUnanswerable questions:")
    for case in results["cases"]:
        label = "OK " if case["honest"] else "BAD"
        print(f"[{label}] {case['question']}\n      -> {case['answer']}\n")
    print(f"\nHonest refusals: {results['honest']}/{results['total']} = {results['honest'] / results['total']:.2f}")


def evaluate_unanswerable(agent) -> dict:
    judge = get_chat_model()
    results = unanswerable_results(agent, judge)
    print_unanswerable_report(results)
    return results


def main() -> None:
    agent = build_agent()
    evaluate_tool_routing(agent)
    evaluate_unanswerable(agent)


if __name__ == "__main__":
    main()
