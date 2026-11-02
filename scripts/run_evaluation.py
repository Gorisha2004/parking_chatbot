"""Single reproducible command for the Stage 1 evaluation.

Runs retrieval, generation and agent evaluation once each, prints progress,
and writes the combined results to evaluation/REPORT.md.
"""
from parking_chatbot.evaluation.report import main

if __name__ == "__main__":
    main()
