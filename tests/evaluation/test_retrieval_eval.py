from parking_chatbot.evaluation.retrieval_eval import (
    low_recall_cases,
    precision_at_k,
    precision_recall_table,
    recall_at_k,
)


def test_precision_at_k_all_relevant():
    assert precision_at_k(["faq", "faq"], ["faq"]) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k(["facilities"], ["faq"]) == 0.0


def test_precision_at_k_of_empty_retrieval_is_zero():
    assert precision_at_k([], ["faq"]) == 0.0


def test_recall_at_k_finds_all_expected_sources():
    assert recall_at_k(["faq", "facilities"], ["faq", "facilities"]) == 1.0


def test_recall_at_k_partial_match():
    assert recall_at_k(["faq"], ["faq", "facilities"]) == 0.5


def test_precision_recall_table_has_one_row_per_k_value():
    results = [{"sources": ["faq", "faq", "faq"], "expected": ["faq"]}]
    table = precision_recall_table(results)
    assert [row["k"] for row in table] == [1, 3, 5]
    assert all(row["precision"] == 1.0 for row in table)


def test_low_recall_cases_filters_out_fully_recalled_cases():
    results = [
        {"question": "q1", "sources": ["faq"], "expected": ["faq"]},
        {"question": "q2", "sources": ["faq"], "expected": ["faq", "facilities"]},
    ]
    low = low_recall_cases(results)
    assert [r["question"] for r in low] == ["q2"]
