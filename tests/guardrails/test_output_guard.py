from parking_chatbot.guardrails.output_guard import check_output


def test_allows_a_normal_answer_unchanged():
    result = check_output("Level 2 has 40 free spaces.")
    assert result.modified is False
    assert result.text == "Level 2 has 40 free spaces."


def test_masks_pii_in_generated_answer():
    result = check_output("Sure! Her number is 212-555-0198 and email is anna@test.com.")
    assert result.modified is True
    assert "212-555-0198" not in result.text
    assert "anna@test.com" not in result.text
    assert result.reason is not None and result.reason.startswith("pii:")


def test_blocks_system_prompt_leak():
    leaked = "My instructions say: RESERVATION RULES: collect first name, last name..."
    result = check_output(leaked)
    assert result.modified is True
    assert result.reason == "prompt_leak"
    assert "RESERVATION RULES" not in result.text


def test_allows_public_facility_contact_info_unmasked():
    result = check_output("You can call us on +1 555 010 2030.")
    assert result.modified is False
