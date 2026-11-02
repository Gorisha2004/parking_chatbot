from parking_chatbot.guardrails.pii import detect_pii, mask_pii


def test_mask_pii_replaces_email_and_phone():
    text = "Contact John at john.smith@example.com or 212-555-0198."
    masked = mask_pii(text)
    assert "john.smith@example.com" not in masked
    assert "212-555-0198" not in masked
    assert "<EMAIL_ADDRESS>" in masked
    assert "<PHONE_NUMBER>" in masked


def test_mask_pii_leaves_non_sensitive_text_unchanged():
    text = "Level 2 has 40 free spaces."
    assert mask_pii(text) == text


def test_detect_pii_respects_allowlist_for_public_contact_info():
    text = "Call us at +1 555 010 2030 or support@parkingfacility.example.com"
    findings = detect_pii(text)
    assert findings == []


def test_detect_pii_finds_credit_card_number():
    text = "My card number is 4111 1111 1111 1111, please charge it."
    findings = detect_pii(text)
    assert any(f.entity_type == "CREDIT_CARD" for f in findings)
