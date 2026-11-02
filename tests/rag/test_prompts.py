import re

from parking_chatbot.chatbot.tools import ALL_TOOLS
from parking_chatbot.rag.prompts import AGENT_SYSTEM_PROMPT, RAG_PROMPT


def test_every_tool_named_in_the_agent_prompt_actually_exists():
    tool_names = {t.name for t in ALL_TOOLS}
    referenced = set(
        re.findall(r"\b(?:search|lookup|update|add|remove|show|submit|cancel|check)_[a-z_]+\b", AGENT_SYSTEM_PROMPT)
    )
    assert referenced, "expected the prompt to reference tools by name"
    assert referenced <= tool_names, f"prompt references unknown tools: {referenced - tool_names}"


def test_agent_system_prompt_states_reservation_confirmation_rule():
    assert "submit_reservation" in AGENT_SYSTEM_PROMPT
    assert "confirm" in AGENT_SYSTEM_PROMPT.lower()


def test_agent_system_prompt_forbids_other_users_data():
    assert "other users" in AGENT_SYSTEM_PROMPT.lower()


def test_agent_system_prompt_forbids_stretching_the_valuables_rule_to_vehicle_damage():
    lowered = AGENT_SYSTEM_PROMPT.lower()
    assert "not responsible for vehicle damage" in lowered
    assert "valuables" in lowered


def test_agent_system_prompt_explains_every_booking_status_in_plain_words():
    lowered = AGENT_SYSTEM_PROMPT.lower()
    assert "check_booking_status" in AGENT_SYSTEM_PROMPT
    for status in ("pending", "approved", "refused", "expired"):
        assert status in lowered


def test_agent_system_prompt_refuses_to_cancel_a_submitted_booking_without_confirmation():
    lowered = AGENT_SYSTEM_PROMPT.lower()
    assert "cannot be cancelled" in lowered or "can never be cancelled" in lowered
    assert "no confirmation" in lowered
    assert "never invent contact details" in lowered
    assert "never call cancel_reservation for this" in lowered


def test_rag_prompt_formats_context_and_question():
    messages = RAG_PROMPT.format_messages(context="Gate A is the entrance.", question="Where is the entrance?")
    rendered = "\n".join(m.content for m in messages)
    assert "Gate A is the entrance." in rendered
    assert "Where is the entrance?" in rendered
