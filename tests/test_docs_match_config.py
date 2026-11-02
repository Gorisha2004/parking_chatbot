"""The static docs quote some limits as plain text; fail loudly if they drift from config."""
from parking_chatbot import config

BOOKING_DOC = (config.STATIC_DATA_DIR / "booking_process.md").read_text(encoding="utf-8")
FAQ_DOC = (config.STATIC_DATA_DIR / "faq.md").read_text(encoding="utf-8")


def test_docs_quote_the_configured_group_size():
    expected = f"up to {config.MAX_VEHICLES_PER_BOOKING} vehicles"
    assert expected in BOOKING_DOC
    assert expected in FAQ_DOC


def test_docs_quote_the_configured_advance_booking_window():
    assert f"({config.MAX_ADVANCE_DAYS} days)" in BOOKING_DOC
