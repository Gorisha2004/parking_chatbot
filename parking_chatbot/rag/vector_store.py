from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from parking_chatbot import config
from parking_chatbot.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Build the vector store used for retrieval.

    Chroma is used because it is embedded (no extra infrastructure to run or
    pay for) and persists to a local directory, which fits a single-user CLI
    project. The return type is the LangChain VectorStore abstraction, so a
    hosted store (Milvus, Pinecone, Weaviate) could replace it here without
    changing any caller.

    Cached to one instance: the agent can call search_parking_info more than
    once in a single turn (one per topic in the user's question), and those
    calls run concurrently on separate threads. Building a fresh Chroma client
    - which opens its own connection to the same persistent directory - from
    two threads at once races inside Chroma's startup check and raises
    AttributeError or ValueError. Reusing one client avoids the race; reads
    through an already-built client are safe to share across threads.
    """
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )