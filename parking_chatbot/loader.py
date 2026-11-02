from pathlib import Path
from langchain_core.documents import Document
from parking_chatbot import config

def load_static_documents(data_dir: Path = config.STATIC_DATA_DIR) -> list[Document]:
    """Read every .md file in data_dir and return them as Document objects."""
    # data_dir: Path = config.STATIC_DATA_DIR: the function takes one input, and the : Path is a type hint saying “this should be a Path”. The = config.STATIC_DATA_DIR is a default value, so calling load_static_documents() with no input uses our static folder.
    documents: list[Document] = []  
    for file_path in sorted(data_dir.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        metadata = {"source": file_path.name}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents