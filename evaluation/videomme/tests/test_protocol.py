from evaluation.videomme.protocol import answer_letter, completed_ids, prompt_for, summarize


def test_answer_letter():
    assert answer_letter("Answer: b") == "B"
    assert answer_letter("C.") == "C"
    assert answer_letter("The final choice is D") == "D"
    assert answer_letter("no option") is None


def test_prompt_contains_question_and_options():
    prompt = prompt_for({"question": "What happens?", "options": ["A. Runs", "B. Sleeps"]})
    assert "What happens?" in prompt
    assert "A. Runs" in prompt


def test_resume_and_summary_use_latest_prediction():
    predictions = [
        {"question_id": "1", "prediction": "A", "answer": "B", "error": None, "task_type": "OCR"},
        {"question_id": "1", "prediction": "B", "answer": "B", "error": None, "task_type": "OCR"},
        {"question_id": "2", "answer": "A", "error": "timeout"},
    ]
    assert completed_ids(predictions) == {"1"}
    summary = summarize(predictions)
    assert summary["overall"] == {"accuracy": 1.0, "n": 1}
    assert summary["errors"] == 1
