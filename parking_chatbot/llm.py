from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from parking_chatbot import config


def get_chat_model() -> BaseChatModel:
    """Build the chat model used for generation and agent reasoning.

    Returns the LangChain BaseChatModel abstraction (not the concrete Azure
    class) so callers depend on an interface and the provider can be swapped
    here without touching any caller.
    """
    return AzureChatOpenAI(
        azure_endpoint=config.DIAL_ENDPOINT,
        api_key=config.DIAL_API_KEY,
        api_version=config.DIAL_API_VERSION,
        azure_deployment=config.CHAT_MODEL,
        temperature=0,
    )