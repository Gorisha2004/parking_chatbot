from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from parking_chatbot import config


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)