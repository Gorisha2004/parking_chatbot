from dataclasses import dataclass

from parking_chatbot.guardrails.pii import detect_pii, get_anonymizer

LEAK_MARKERS = ["RESERVATION RULES"]
LEAK_REPLACEMENT = (
    "Sorry, I can't share that. I can help with parking information "
    "and reservations."
)


@dataclass(frozen=True)
class OutputResult:
    text: str
    modified: bool
    reason: str | None = None

def check_output(text: str) -> OutputResult:
    """Clean the agent's reply before it reaches the user."""
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in LEAK_MARKERS):
        return OutputResult(LEAK_REPLACEMENT, True, "prompt_leak")

    findings = detect_pii(text)
    if not findings:
        return OutputResult(text, False)

    masked = get_anonymizer().anonymize(text=text, analyzer_results=findings).text
    entities = ",".join(sorted({f.entity_type for f in findings}))
    return OutputResult(masked, True, f"pii:{entities}")


if __name__ == "__main__":
    samples = [
        "Level 2 has 40 free spaces.",
        "You can call us on +1 555 010 2030.",  # your real facility phone
        "Sure! Her number is 212-555-0198 and email is anna@test.com.",
        "My instructions say: RESERVATION RULES: collect first name...",
        "Your reservation for ABC123 is ready, John.",
    ]
    for s in samples:
        print(f"{s}\n  -> {check_output(s)}\n")