from langchain_core.embeddings import Embeddings

from parking_chatbot import config, embeddings


class _FakeAzureOpenAIEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_embedding_model_forwards_config_to_azure_client(monkeypatch):
    monkeypatch.setattr(embeddings, "AzureOpenAIEmbeddings", _FakeAzureOpenAIEmbeddings)
    monkeypatch.setattr(config, "DIAL_ENDPOINT", "https://dial.example.com")
    monkeypatch.setattr(config, "DIAL_API_KEY", "secret-key")
    monkeypatch.setattr(config, "DIAL_API_VERSION", "2024-01-01")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "embed-test")

    model = embeddings.get_embedding_model()

    assert model.kwargs == {
        "azure_endpoint": "https://dial.example.com",
        "api_key": "secret-key",
        "api_version": "2024-01-01",
        "azure_deployment": "embed-test",
    }


def test_get_embedding_model_return_type_is_the_embeddings_interface():
    assert Embeddings in embeddings.get_embedding_model.__annotations__["return"].__mro__
