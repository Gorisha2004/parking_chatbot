from langchain_core.language_models.chat_models import BaseChatModel

from parking_chatbot import config, llm


class _FakeAzureChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_chat_model_forwards_config_to_azure_client(monkeypatch):
    monkeypatch.setattr(llm, "AzureChatOpenAI", _FakeAzureChatOpenAI)
    monkeypatch.setattr(config, "DIAL_ENDPOINT", "https://dial.example.com")
    monkeypatch.setattr(config, "DIAL_API_KEY", "secret-key")
    monkeypatch.setattr(config, "DIAL_API_VERSION", "2024-01-01")
    monkeypatch.setattr(config, "CHAT_MODEL", "gpt-test")

    model = llm.get_chat_model()

    assert model.kwargs == {
        "azure_endpoint": "https://dial.example.com",
        "api_key": "secret-key",
        "api_version": "2024-01-01",
        "azure_deployment": "gpt-test",
        "temperature": 0,
    }


def test_get_chat_model_return_type_is_the_base_chat_model_interface():
    assert BaseChatModel in llm.get_chat_model.__annotations__["return"].__mro__
