import threading

import pytest
from langchain_core.vectorstores import VectorStore

from parking_chatbot import config
from parking_chatbot.rag import vector_store as vector_store_module


class _FakeChroma:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def clear_vector_store_cache():
    """get_vector_store is cached (see its docstring for why); without clearing the
    cache between tests, whichever test runs first would decide every later test's
    "fresh" instance, real or fake."""
    vector_store_module.get_vector_store.cache_clear()
    yield
    vector_store_module.get_vector_store.cache_clear()


def test_get_vector_store_forwards_config(monkeypatch):
    monkeypatch.setattr(vector_store_module, "Chroma", _FakeChroma)
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", "test_chroma_dir")
    fake_embeddings = object()
    monkeypatch.setattr(
        vector_store_module, "get_embedding_model", lambda: fake_embeddings
    )

    store = vector_store_module.get_vector_store()

    assert store.kwargs == {
        "collection_name": "test_collection",
        "embedding_function": fake_embeddings,
        "persist_directory": "test_chroma_dir",
    }


def test_get_vector_store_return_type_is_the_vector_store_interface():
    annotation = vector_store_module.get_vector_store.__annotations__["return"]
    assert VectorStore in annotation.__mro__


def test_get_vector_store_returns_the_same_instance_every_call(monkeypatch):
    monkeypatch.setattr(vector_store_module, "Chroma", _FakeChroma)
    monkeypatch.setattr(vector_store_module, "get_embedding_model", lambda: object())

    assert vector_store_module.get_vector_store() is vector_store_module.get_vector_store()


def test_concurrent_calls_build_the_client_only_once(monkeypatch):
    """Regression test: the agent can call search_parking_info twice in one turn (one
    per topic), and LangGraph runs those calls on separate threads. Building a fresh
    Chroma client per call raced inside Chroma's own startup check and raised
    AttributeError/ValueError. Caching must mean only one client is ever built."""
    build_count = 0

    class _CountingChroma:
        def __init__(self, **kwargs):
            nonlocal build_count
            build_count += 1

    monkeypatch.setattr(vector_store_module, "Chroma", _CountingChroma)
    monkeypatch.setattr(vector_store_module, "get_embedding_model", lambda: object())

    results = []
    threads = [threading.Thread(target=lambda: results.append(vector_store_module.get_vector_store())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert build_count == 1
    assert len({id(store) for store in results}) == 1
