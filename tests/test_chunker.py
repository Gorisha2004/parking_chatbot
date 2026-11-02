from langchain_core.documents import Document

from parking_chatbot import config
from parking_chatbot.chunker import split_documents


def test_short_document_is_not_split():
    documents = [Document(page_content="Short text.", metadata={"source": "a.md"})]
    chunks = split_documents(documents)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_long_document_is_split_into_multiple_overlapping_chunks():
    long_text = ("Parking information. " * 50).strip()
    documents = [Document(page_content=long_text, metadata={"source": "big.md"})]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= config.CHUNK_SIZE for chunk in chunks)


def test_chunk_metadata_preserves_source():
    long_text = "Parking information. " * 50
    documents = [Document(page_content=long_text, metadata={"source": "big.md"})]

    chunks = split_documents(documents)

    assert all(chunk.metadata["source"] == "big.md" for chunk in chunks)
