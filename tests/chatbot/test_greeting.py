import pytest

from parking_chatbot import config
from parking_chatbot.chatbot import agent as agent_module
from parking_chatbot.chatbot.greeting import is_greeting
from parking_chatbot.rag.prompts import AGENT_SYSTEM_PROMPT, GREETING_MESSAGE


@pytest.mark.parametrize(
    "text", ["Hi", "hi!", "Hii", "  hello ", "Hello there!", "hey bot", "Good morning", "good evening!!", "Namaste"]
)
def test_plain_greetings_are_recognised(text):
    assert is_greeting(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Hi, what are your prices?",
        "hello I want to book a space",
        "Is there space on level 3?",
        "history of the parking",
        "",
    ],
)
def test_messages_with_a_real_question_are_not_greetings(text):
    assert is_greeting(text) is False


def test_greeting_introduces_the_bot_its_services_and_how_to_start():
    assert "virtual assistant of CityPark Central" in GREETING_MESSAGE
    for service in ("opening hours", "prices", "free spaces", "group booking", "pass ID"):
        assert service in GREETING_MESSAGE
    assert "To get started" in GREETING_MESSAGE
    assert f"up to {config.MAX_VEHICLES_PER_BOOKING} vehicles" in GREETING_MESSAGE


class _AgentThatMustNotBeCalled:
    def invoke(self, payload, config):
        raise AssertionError("a plain greeting must not reach the LLM")


def test_chat_answers_a_greeting_with_the_fixed_introduction_without_the_llm():
    answer, tools = agent_module.chat(_AgentThatMustNotBeCalled(), "Hi", thread_id="greet")
    assert answer == GREETING_MESSAGE
    assert tools == []


def test_agent_prompt_carries_the_same_introduction_for_greetings_the_pattern_misses():
    assert GREETING_MESSAGE in AGENT_SYSTEM_PROMPT
