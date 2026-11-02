from langchain_core.documents import Document

from parking_chatbot.rag import chain


class _FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.last_call = None

    def similarity_search(self, question, k):
        self.last_call = (question, k)
        return self.docs[:k]


def test_format_context_joins_pages_with_blank_line():
    docs = [Document(page_content="A"), Document(page_content="B")]
    assert chain.format_context(docs) == "A\n\nB"


def test_format_context_of_empty_list_is_empty_string():
    assert chain.format_context([]) == ""


def test_retrieve_passes_question_and_k_to_vector_store(monkeypatch):
    docs = [Document(page_content=f"doc {i}") for i in range(5)]
    fake_store = _FakeVectorStore(docs)
    monkeypatch.setattr(chain, "get_vector_store", lambda: fake_store)

    result = chain.retrieve("where is it?", k=2)

    assert fake_store.last_call == ("where is it?", 2)
    assert result == docs[:2]


def test_generate_answer_invokes_chain_with_context_and_question(monkeypatch):
    docs = [Document(page_content="Entrance is on Main Street.")]

    class _FakeRunnable:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            self.payload = payload
            return "Entrance is on Main Street."

    fake_runnable = _FakeRunnable()
    monkeypatch.setattr(chain, "RAG_PROMPT", fake_runnable)
    monkeypatch.setattr(chain, "get_chat_model", lambda: object())

    answer = chain.generate_answer("Where is the entrance?", docs)

    assert answer == "Entrance is on Main Street."
    assert fake_runnable.payload == {
        "context": "Entrance is on Main Street.",
        "question": "Where is the entrance?",
    }
