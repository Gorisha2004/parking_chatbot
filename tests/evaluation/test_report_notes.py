from parking_chatbot.evaluation import report_notes


def test_setup_template_fills_in_every_placeholder():
    text = report_notes.SETUP_TEMPLATE.format(
        chat_model="m", embedding_model="e", chunk_size=1, chunk_overlap=2,
        collection_name="c", top_k=3, doc_count=4, chunk_count=5,
        retrieval_count=6, unanswerable_count=7, routing_count=8,
    )
    assert "{" not in text and "}" not in text


def test_limitations_template_fills_in_every_placeholder():
    text = report_notes.LIMITATIONS_TEMPLATE.format(
        top_k=3, doc_count=4, chunk_count=5, retrieval_count=6,
        routing_count=8, unanswerable_count=7, honesty_judgments=21, honesty_runs=3,
    )
    assert "{" not in text and "}" not in text


def test_metric_definitions_and_before_correction_text_need_no_placeholders():
    assert "{" not in report_notes.METRIC_DEFINITIONS
    assert "{" not in report_notes.BEFORE_CORRECTION_INTRO
    assert "0.46" in report_notes.BEFORE_CORRECTION_INTRO  # the frozen pre-correction numbers
