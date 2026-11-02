import importlib.util

from langchain_core.documents import Document

from parking_chatbot import config


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, config.PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest = _load_script("ingest")


def test_build_chunk_ids_numbers_chunks_per_source():
    chunks = [
        Document(page_content="a", metadata={"source": "faq.md"}),
        Document(page_content="b", metadata={"source": "faq.md"}),
        Document(page_content="c", metadata={"source": "location.md"}),
    ]
    assert ingest.build_chunk_ids(chunks) == ["faq.md-0", "faq.md-1", "location.md-0"]


def test_build_chunk_ids_are_deterministic_across_runs():
    chunks = [Document(page_content="a", metadata={"source": "faq.md"})] * 3
    assert ingest.build_chunk_ids(chunks) == ingest.build_chunk_ids(chunks)


def test_main_stores_every_chunk_with_its_id(monkeypatch):
    stored = {}

    class _FakeStore:
        def add_documents(self, documents, ids):
            stored["documents"], stored["ids"] = documents, ids

    docs = [Document(page_content="x" * 10, metadata={"source": "faq.md"})]
    monkeypatch.setattr(ingest, "load_static_documents", lambda: docs)
    monkeypatch.setattr(ingest, "get_vector_store", lambda: _FakeStore())

    ingest.main()

    assert stored["ids"] == ["faq.md-0"]
    assert len(stored["documents"]) == 1
