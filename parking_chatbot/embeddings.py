from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings

from parking_chatbot import config


def get_embedding_model() -> Embeddings:
    """Build the embedding model used for ingestion and retrieval.

    Returns the LangChain Embeddings abstraction so the vector store and
    callers depend on an interface, not the concrete Azure implementation.
    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=config.DIAL_ENDPOINT,
        api_key=config.DIAL_API_KEY,
        api_version=config.DIAL_API_VERSION,
        azure_deployment=config.EMBEDDING_MODEL,
    )