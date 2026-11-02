"""Hand-labelled evaluation data. Labels are the data/static/*.md files (without .md) that hold the answer."""

# Corrections made after a run showed a label contradicting the source documents. Each was checked
# against data/static/*.md; they are listed in the report so results stay comparable and honest.
LABEL_CORRECTIONS = [
    "Emergency phones question: expected sources changed from contact_info + facilities to contact_info "
    "only; the emergency phones are described in contact_info.md, not in facilities.md.",
    "Office / online question: moved from the unanswerable set to the retrieval set (expected contact_info); "
    "contact_info.md gives the administration office address, so the documents do answer it.",
]

RETRIEVAL_CASES = [

    {"question": "What is the contact phone number?", "expected": ["contact_info"]},
    {"question": "How do I make a reservation?", "expected": ["booking_process"]},
    {"question": "What payment methods do you accept?", "expected": ["payment_methods"]},
    {"question": "Where is the parking located?", "expected": ["location"]},
    {"question": "What are the parking rules?", "expected": ["parking_rules"]},
    {"question": "What facilities are available?", "expected": ["facilities"]},

    {"question": "How to check if my reservation is done?", "expected": ["booking_process"]},
    {"question": "Are there elevators in parking slot for going to different levels?", "expected": ["general_info"]},
    {"question": "Can I pay by card?", "expected": ["payment_methods"]},
    {"question": "Is there any security guard assigned at parking slot?", "expected": ["facilities"]},

    {"question": "What should I do if I lose my parking ticket?", "expected": ["faq"]},
    {"question": "Is my car safe if I park it overnight?", "expected": ["faq"]},

    {"question": "What is the maximum vehicle height allowed in this parking facility?", "expected": ["general_info"]},
    {"question": "How many levels does CityPark Central have?", "expected": ["general_info"]},
    
    {"question": "Do you accept contactless payments like Apple Pay?", "expected": ["payment_methods"]},
    {"question": "What should I do if my vehicle gets damaged inside the parking?", "expected": ["faq"]},
    {"question": "Is overnight parking permitted without a reservation?", "expected": ["parking_rules"]},
    
    {"question": "How can I get to CityPark Central by train and what is near the entrance?", "expected": ["location", "general_info"]},
    {"question": "What emergency contact number should I use and where are emergency phones located?", "expected": ["contact_info"]},
    {"question": "Is there any workplace from where this parking slot is operating or are they working online?", "expected": ["contact_info"]},

    {"question": "Are there parking spaces for disabled people?", "expected": ["facilities"]},
    {"question": "Do you offer electric vehicle charging stations?", "expected": ["facilities"]},
    {"question": "What is the cancellation policy for reservations?", "expected": ["booking_process"]},
    {"question": "How can I reserve a parking space through the chatbot?", "expected": ["booking_process"]},

    {"question": "Can I park a three-wheeler or an auto-rickshaw here?", "expected": ["vehicles"]},
    {"question": "Can I park a truck or a bus?", "expected": ["vehicles"]},
    {"question": "Are CNG or LPG gas-powered cars allowed inside?", "expected": ["vehicles"]},
    {"question": "Can I bring my motorcycle?", "expected": ["vehicles"]},

    {"question": "Can I book several spaces together for a family gathering?", "expected": ["booking_process", "faq"]},
    {"question": "What is a pass ID and how do drivers use it?", "expected": ["booking_process"]},
]

UNANSWERABLE_QUESTIONS = [
    "Do you offer a discount for a yearly subscription?",
    "Is every level assigned for one kind of vehicle parking?",
    "What are the offers present if any customer wants to buy a subscription for the complete year?",
    "Is your parking area registered and authorized to park vehicles?",
    "Is WiFi available in the parking facility?",
    "Can I reserve a specific parking level or spot?",
    "What is your refund policy for damaged vehicles?",
]

TOOL_ROUTING_CASES = [
    {"question": "What are the parking rates or prices?", "expected_tools": {"lookup_prices"}},
    {"question": "Do you offer monthly or weekly subscription plans?", "expected_tools": {"lookup_prices"}},
    {"question": "What are your opening hours?", "expected_tools": {"lookup_working_hours"}},
    {"question": "Is the parking open on Sundays?", "expected_tools": {"lookup_working_hours"}},
    {"question": "How much does it cost per hour?", "expected_tools": {"lookup_prices"}},
    {"question": "What is the monthly rate?", "expected_tools": {"lookup_prices"}},
    {"question": "Are there any free spaces right now?", "expected_tools": {"lookup_availability"}},
    {"question": "Is there space on level 3?", "expected_tools": {"lookup_availability"}},
    {"question": "Where is the parking located?", "expected_tools": {"search_parking_info"}},
    {"question": "Do you have electric vehicle charging?", "expected_tools": {"search_parking_info"}},
    {"question": "What payment methods do you accept?", "expected_tools": {"search_parking_info"}},
    {"question": "What is the phone number for customer support?", "expected_tools": {"search_parking_info"}},
    {"question": "How much is a daily ticket and what are the opening hours?",
     "expected_tools": {"lookup_prices", "lookup_working_hours"}},
    {"question": "Is there space available and how much is it per hour?",
     "expected_tools": {"lookup_availability", "lookup_prices"}},
    {"question": "What emergency contact number should I use and where are emergency phones located?",
     "expected_tools": {"search_parking_info"}},
    {"question": "How much does it cost to park a motorcycle for a day?", "expected_tools": {"lookup_prices"}},
    {"question": "Can I park a truck here?", "expected_tools": {"search_parking_info"}},
    {"question": "Hello!", "expected_tools": set()},
]