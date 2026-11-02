"""Recognise messages that are only a greeting, so they get the fixed introduction."""
import re

_GREETING = re.compile(
    r"^\s*(hi+|hey+|hello+|hiya|howdy|greetings|namaste|good\s+(morning|afternoon|evening|day))"
    r"(\s+(there|bot|assistant|all|everyone|team))?\s*[!.,?]*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """True if the message is nothing but a greeting, e.g. "Hi" or "Good morning!"."""
    return bool(_GREETING.match(text))
