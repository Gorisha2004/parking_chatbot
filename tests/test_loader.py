from parking_chatbot.loader import load_static_documents


def test_loads_one_document_per_markdown_file(tmp_path):
    (tmp_path / "a.md").write_text("Alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("Beta content", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not markdown", encoding="utf-8")

    documents = load_static_documents(tmp_path)

    assert len(documents) == 2
    assert {doc.metadata["source"] for doc in documents} == {"a.md", "b.md"}


def test_document_content_matches_file_content(tmp_path):
    (tmp_path / "only.md").write_text("Hello world", encoding="utf-8")

    documents = load_static_documents(tmp_path)

    assert documents[0].page_content == "Hello world"


def test_empty_directory_returns_no_documents(tmp_path):
    assert load_static_documents(tmp_path) == []
