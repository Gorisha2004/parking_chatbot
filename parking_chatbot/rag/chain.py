from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser 

from parking_chatbot import config
from parking_chatbot.llm import get_chat_model         
from parking_chatbot.rag.prompts import RAG_PROMPT     
from parking_chatbot.rag.vector_store import get_vector_store

def format_context(documents: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)

def retrieve(question: str, k: int = config.TOP_K) -> list[Document]:
    vector_store = get_vector_store()
    return vector_store.similarity_search(question, k=k)

def generate_answer(question: str, documents: list[Document]) -> str:
    chain = RAG_PROMPT | get_chat_model() | StrOutputParser()
    return chain.invoke({"context": format_context(documents), "question": question})

def answer_question(question: str) -> str:
    documents = retrieve(question)
    return generate_answer(question, documents)