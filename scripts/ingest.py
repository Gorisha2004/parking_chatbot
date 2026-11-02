from langchain_core.documents import Document

from parking_chatbot.chunker import split_documents
from parking_chatbot.loader import load_static_documents
from parking_chatbot.rag.vector_store import get_vector_store


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    counters: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        source = chunk.metadata["source"]
        index = counters.get(source, 0)
        ids.append(f"{source}-{index}")
        counters[source] = index + 1
    return ids

def main() -> None:
    documents = load_static_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")
    ids = build_chunk_ids(chunks)
    vector_store = get_vector_store()
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"Stored {len(ids)} chunks in Chroma")


if __name__ == "__main__":
    main()
