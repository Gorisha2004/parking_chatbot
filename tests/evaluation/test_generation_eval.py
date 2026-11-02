from langchain_core.documents import Document

from parking_chatbot.evaluation import generation_eval


def test_evaluate_generation_times_each_case(monkeypatch):
    monkeypatch.setattr(generation_eval, "retrieve", lambda question: [Document(page_content="ctx")])
    monkeypatch.setattr(generation_eval, "generate_answer", lambda question, docs: f"answer to {question}")

    results = generation_eval.evaluate_generation()

    assert len(results) == len(generation_eval.RETRIEVAL_CASES)
    assert all(r["latency"] >= 0 for r in results)
    assert results[0]["answer"].startswith("answer to ")


def test_evaluate_generation_uses_retrieved_docs_as_context(monkeypatch):
    captured = []
    monkeypatch.setattr(generation_eval, "retrieve", lambda question: [Document(page_content="ctx")])
    monkeypatch.setattr(
        generation_eval,
        "generate_answer",
        lambda question, docs: captured.append((question, docs)) or "ok",
    )

    generation_eval.evaluate_generation()

    assert len(captured) == len(generation_eval.RETRIEVAL_CASES)
    assert all(docs[0].page_content == "ctx" for _, docs in captured)
