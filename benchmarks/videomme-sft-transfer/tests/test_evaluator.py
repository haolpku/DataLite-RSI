import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("videomme_evaluator", MODULE)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


def test_scores_overall_and_categories():
    result = EVALUATOR.score_rows([
        {"category": "OCR", "prediction": "a", "answer": "A"},
        {"category": "OCR", "prediction": "C", "answer": "B"},
        {"category": "Counting", "prediction": "D", "answer": "D"},
    ])
    assert result == {"overall": 2 / 3, "categories": {"counting": 1.0, "ocr": 0.5}}


def test_rejects_unknown_categories():
    try:
        EVALUATOR.score_rows([{"category": "Other", "prediction": "A", "answer": "A"}])
    except ValueError as exc:
        assert "unknown Video-MME category" in str(exc)
    else:
        raise AssertionError("expected unknown category to fail")


def test_accepts_harness_task_type_field():
    result = EVALUATOR.score_rows([
        {"task_type": "Action Recognition", "prediction": "A", "answer": "A"},
    ])
    assert result["categories"] == {"action_recognition": 1.0}
