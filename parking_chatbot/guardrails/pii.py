from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

SENSITIVE_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN"]
SCORE_THRESHOLD = 0.4
PUBLIC_ALLOWLIST = ["+1 555 010 2030", "support@parkingfacility.example.com"]

@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()

def detect_pii(text: str) -> list:
    """Return Presidio findings for sensitive entities in the text."""
    return get_analyzer().analyze(
        text=text,
        language="en",
        entities=SENSITIVE_ENTITIES,
        score_threshold=SCORE_THRESHOLD,
        allow_list=PUBLIC_ALLOWLIST,
    )


def mask_pii(text: str) -> str:
    """Replace sensitive data with placeholders like <EMAIL_ADDRESS>."""
    findings = detect_pii(text)
    if not findings:
        return text
    return get_anonymizer().anonymize(text=text, analyzer_results=findings).text


if __name__ == "__main__":
    samples = [
        "Call John on 212-555-0198 or mail john@test.com",
        "My card is 4111 1111 1111 1111",
        "Contact us at +1 555 010 2030",  # use your real phone here
        "Level 2 has 40 free spaces",
    ]
    for s in samples:
        print(f"{s}\n  -> {mask_pii(s)}\n")