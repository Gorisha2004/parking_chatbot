from parking_chatbot import config
from parking_chatbot.evaluation.dataset import (
    RETRIEVAL_CASES,
    TOOL_ROUTING_CASES,
    UNANSWERABLE_QUESTIONS,
)


def test_every_retrieval_case_has_a_question_and_expected_sources():
    assert len(RETRIEVAL_CASES) > 0
    for case in RETRIEVAL_CASES:
        assert case["question"].strip()
        assert len(case["expected"]) > 0


def test_every_tool_routing_case_has_a_question_and_expected_tools_set():
    assert len(TOOL_ROUTING_CASES) > 0
    for case in TOOL_ROUTING_CASES:
        assert case["question"].strip()
        assert isinstance(case["expected_tools"], set)


def test_unanswerable_questions_are_non_empty_strings():
    assert len(UNANSWERABLE_QUESTIONS) > 0
    assert all(question.strip() for question in UNANSWERABLE_QUESTIONS)


def test_a_question_is_never_both_answerable_and_unanswerable():
    retrieval_questions = {case["question"] for case in RETRIEVAL_CASES}
    assert not retrieval_questions & set(UNANSWERABLE_QUESTIONS)


def test_every_expected_source_is_a_real_document():
    documents = {path.stem for path in config.STATIC_DATA_DIR.glob("*.md")}
    for case in RETRIEVAL_CASES:
        assert set(case["expected"]) <= documents, case["question"]
