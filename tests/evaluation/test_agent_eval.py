from parking_chatbot.evaluation import agent_eval


class _FakeJudge:
    def __init__(self, verdict: str):
        self.verdict = verdict

    def invoke(self, prompt):
        class _Reply:
            def __init__(self, content):
                self.content = content

        return _Reply(self.verdict)


def test_judge_honesty_true_for_honest_verdict():
    assert agent_eval.judge_honesty(_FakeJudge("HONEST"), "q", "a") is True


def test_judge_honesty_false_for_invented_verdict():
    assert agent_eval.judge_honesty(_FakeJudge("INVENTED"), "q", "a") is False


def test_judge_honesty_is_case_and_whitespace_tolerant():
    assert agent_eval.judge_honesty(_FakeJudge("  honest\n"), "q", "a") is True


def test_judge_prompt_tells_the_judge_not_to_penalize_true_supplementary_facts():
    lowered = agent_eval.JUDGE_PROMPT.lower()
    assert "does not make it invented" in lowered


def test_tool_routing_results_counts_matches_and_mismatches(monkeypatch):
    responses = iter([
        ("price answer", {"lookup_prices"}, 0.1),
        ("wrong tool used", {"lookup_working_hours"}, 0.2),
    ])
    monkeypatch.setattr(agent_eval, "TOOL_ROUTING_CASES", [
        {"question": "What are your prices?", "expected_tools": {"lookup_prices"}},
        {"question": "Is there space?", "expected_tools": {"lookup_availability"}},
    ])
    monkeypatch.setattr(agent_eval, "ask", lambda agent, question: next(responses))

    results = agent_eval.tool_routing_results(agent=object())

    assert results["correct"] == 1
    assert results["total"] == 2
    assert len(results["failures"]) == 1
    assert results["failures"][0]["expected"] == ["lookup_availability"]
    assert results["failures"][0]["got"] == ["lookup_working_hours"]


def _run(*verdicts):
    cases = [{"question": f"q{i}", "answer": f"answer {i} {'ok' if v else 'made up'}", "honest": v}
             for i, v in enumerate(verdicts)]
    return {"honest": sum(verdicts), "total": len(verdicts), "cases": cases}


def test_summarize_honesty_runs_reports_rate_per_run_and_range():
    summary = agent_eval.summarize_honesty_runs([_run(True, True), _run(True, False), _run(False, False)])
    assert summary["rates"] == [1.0, 0.5, 0.0]
    assert (summary["min"], summary["max"], summary["mean"]) == (0.0, 1.0, 0.5)


def test_summarize_honesty_runs_counts_honest_runs_per_question_and_shows_an_invented_reply():
    summary = agent_eval.summarize_honesty_runs([_run(True, True), _run(True, False), _run(True, True)])
    first, second = summary["questions"]
    assert (first["question"], first["honest_runs"], first["runs"]) == ("q0", 3, 3)
    assert first["example_answer"] == "answer 0 ok"
    assert (second["honest_runs"], second["example_answer"]) == (2, "answer 1 made up")


def test_unanswerable_results_counts_honest_replies(monkeypatch):
    monkeypatch.setattr(agent_eval, "UNANSWERABLE_QUESTIONS", ["Do you offer yearly discounts?"])
    monkeypatch.setattr(agent_eval, "ask", lambda agent, question: ("I don't have that information.", set(), 0.1))

    results = agent_eval.unanswerable_results(agent=object(), judge=_FakeJudge("HONEST"))

    assert results == {
        "honest": 1,
        "total": 1,
        "cases": [
            {"question": "Do you offer yearly discounts?", "answer": "I don't have that information.", "honest": True}
        ],
    }
