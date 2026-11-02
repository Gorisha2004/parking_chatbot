from parking_chatbot.evaluation import report
from parking_chatbot.evaluation.dataset import LABEL_CORRECTIONS


def _patch_evaluations(monkeypatch, honesty_runs=None):
    """Replace every live evaluation (and the local chunk count) with canned results;
    honesty runs are returned in order. No DIAL call and no real Chroma read happens."""
    retrieval_results = [
        {"question": "Where is it?", "expected": ["location"], "sources": ["location"] * 5, "latency": 0.05},
    ]
    generation_results = [{"question": "Where is it?", "answer": "On Main Street.", "latency": 0.5}]
    routing = {"correct": 1, "total": 1, "latencies": [0.7], "failures": []}
    honest = {
        "honest": 1,
        "total": 1,
        "cases": [{"question": "Discounts?", "answer": "I don't know.", "honest": True}],
    }
    runs = iter(honesty_runs or [honest] * report.HONESTY_RUNS)

    monkeypatch.setattr(report.retrieval_eval, "evaluate_retrieval", lambda: retrieval_results)
    monkeypatch.setattr(report.generation_eval, "evaluate_generation", lambda: generation_results)
    monkeypatch.setattr(report, "build_agent", lambda: object())
    monkeypatch.setattr(report.agent_eval, "tool_routing_results", lambda agent: routing)
    monkeypatch.setattr(report.agent_eval, "get_chat_model", lambda: object())
    monkeypatch.setattr(report.agent_eval, "unanswerable_results", lambda agent, judge: next(runs))
    monkeypatch.setattr(report, "_chunk_count", lambda: 50)


def test_build_report_contains_every_required_section(monkeypatch):
    _patch_evaluations(monkeypatch)

    text = report.build_report()

    for heading in (
        "## Setup", "## Method", "### Precision@K and Recall@K, precisely", "## Dataset corrections",
        "## Retrieval quality", "## Latency", "## Tool-routing accuracy",
        "## Honesty on unanswerable questions", "## Limitations", "## Conclusions",
    ):
        assert heading in text


def test_setup_section_uses_live_config_and_local_counts(monkeypatch):
    _patch_evaluations(monkeypatch)
    monkeypatch.setattr(report.config, "CHAT_MODEL", "test-chat-model")
    monkeypatch.setattr(report.config, "EMBEDDING_MODEL", "test-embed-model")

    text = report.build_report()

    assert "`test-chat-model`" in text
    assert "`test-embed-model`" in text
    assert "50 chunks ingested" in text


def test_build_report_reflects_actual_metrics(monkeypatch):
    _patch_evaluations(monkeypatch)

    text = report.build_report()

    assert "1/1 = 1.00" in text
    assert text.count("Every case achieved full Recall@5.") == 2  # before/after section + main table


def test_report_lists_every_label_correction_and_the_frozen_before_numbers(monkeypatch):
    _patch_evaluations(monkeypatch)

    text = report.build_report()

    for correction in LABEL_CORRECTIONS:
        assert correction in text
    assert "0.46 | 0.98" in text  # frozen pre-correction Recall@5, never recomputed
    assert "After correction (1 cases, current run):" in text


def test_conclusions_flag_low_honesty_and_mention_precision_at_1(monkeypatch):
    invented = {
        "honest": 0,
        "total": 1,
        "cases": [{"question": "Discounts?", "answer": "Made up answer.", "honest": False}],
    }
    _patch_evaluations(monkeypatch, honesty_runs=[invented, invented, invented])

    text = report.build_report()

    assert "sometimes invents answers" in text
    assert "Precision@1 = 1.00" in text  # single-case retrieval fixture has Precision@1 = 1.00
    assert report.report_notes.HONESTY_CAVEAT in text


def test_report_runs_the_honesty_check_several_times_and_shows_the_range(monkeypatch):
    invented = {
        "honest": 0,
        "total": 1,
        "cases": [{"question": "Discounts?", "answer": "Yes, 20% off | yearly\nplans!", "honest": False}],
    }
    honest = {"honest": 1, "total": 1, "cases": [{"question": "Discounts?", "answer": "I don't know.", "honest": True}]}
    _patch_evaluations(monkeypatch, honesty_runs=[honest, invented, honest])

    text = report.build_report()

    assert "per run: 1.00, 0.00, 1.00 (range 0.00–1.00)" in text
    assert "| Discounts? | 2/3 | Yes, 20% off \\| yearly plans! |" in text


def test_table_cell_flattens_newlines_escapes_pipes_and_truncates():
    assert report._table_cell("a |\n b") == "a \\| b"
    long_cell = report._table_cell("x" * (report.MAX_REPLY_CHARS + 50))
    assert len(long_cell) == report.MAX_REPLY_CHARS
    assert long_cell.endswith("…")


def test_chunk_count_reads_the_local_collection_without_patching(monkeypatch):
    class _FakeCollection:
        def count(self):
            return 42

    class _FakeStore:
        _collection = _FakeCollection()

    monkeypatch.setattr(report, "get_vector_store", lambda: _FakeStore())
    assert report._chunk_count() == 42


def test_main_writes_report_to_disk_creating_missing_folder(monkeypatch, tmp_path):
    _patch_evaluations(monkeypatch)
    report_path = tmp_path / "evaluation" / "REPORT.md"
    monkeypatch.setattr(report, "REPORT_PATH", report_path)

    report.main()

    assert report_path.exists()
    assert "# Stage 1 Evaluation Report" in report_path.read_text(encoding="utf-8")
