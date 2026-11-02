"""Acceptance #4: sensitive data in the source data never reaches the user."""
from dataclasses import dataclass

import pytest

from parking_chatbot import config
from parking_chatbot.chatbot import agent as agent_module
from parking_chatbot.guardrails.pii import detect_pii

STATIC_DOCS = sorted(config.STATIC_DATA_DIR.glob("*.md"))


@pytest.mark.parametrize("doc", STATIC_DOCS, ids=lambda p: p.name)
def test_static_source_documents_contain_no_sensitive_data(doc):
    findings = detect_pii(doc.read_text(encoding="utf-8"))
    assert findings == [], f"{doc.name} contains sensitive data: {findings}"


@dataclass
class _Message:
    type: str
    content: str
    name: str | None = None


class _AgentEchoingRetrievedChunk:
    """Simulates the agent repeating a retrieved chunk that contains private data."""

    def __init__(self, retrieved_chunk: str):
        self.retrieved_chunk = retrieved_chunk

    def invoke(self, payload, config):
        return {
            "messages": [
                _Message("human", payload["messages"][0]["content"]),
                _Message("tool", self.retrieved_chunk, name="search_parking_info"),
                _Message("ai", f"Here is what I found: {self.retrieved_chunk}"),
            ]
        }


def test_private_data_in_a_retrieved_chunk_is_masked_before_reaching_the_user():
    leaked = "Reservation by Maria Lopez, phone 212-555-0198, email maria.lopez@gmail.com"
    agent = _AgentEchoingRetrievedChunk(leaked)

    answer, _ = agent_module.chat(agent, "Tell me about the booking process", thread_id="leak-test")

    assert "212-555-0198" not in answer
    assert "maria.lopez@gmail.com" not in answer


def test_public_facility_contacts_are_still_shown_to_the_user():
    agent = _AgentEchoingRetrievedChunk("Support: +1 555 010 2030, support@parkingfacility.example.com")

    answer, _ = agent_module.chat(agent, "How do I contact support?", thread_id="public-test")

    assert "+1 555 010 2030" in answer
    assert "support@parkingfacility.example.com" in answer
