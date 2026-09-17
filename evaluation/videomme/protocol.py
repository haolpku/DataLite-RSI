"""Pure-Python Video-MME prompt, answer, and aggregation utilities."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def answer_letter(text: str | None) -> str | None:
    """Extract a multiple-choice answer letter from a model response."""
    value = (text or "").strip().upper()
    explicit = re.search(r"(?:ANSWER|OPTION|CHOICE|FINAL)\s*[:：]?\s*[*`#\[]?\s*([ABCD])\b", value)
    if explicit:
        return explicit.group(1)
    initial = re.match(r"\s*[*`#\[]?\s*([ABCD])[.)：:]", value)
    if initial:
        return initial.group(1)
    letters = re.findall(r"\b([ABCD])\b", value)
    return letters[-1] if letters else None


def prompt_for(row: dict[str, Any]) -> str:
    options = "\n".join(str(option) for option in row.get("options", []))
    return (
        "Watch the video and answer the multiple-choice question. "
        "Return exactly one uppercase letter: A, B, C, or D.\n\n"
        f"Question: {row['question']}\n{options}\n\nAnswer:"
    )


def completed_ids(predictions: list[dict[str, Any]]) -> set[str]:
    """Return question IDs with a successful latest prediction."""
    latest = {str(item["question_id"]): item for item in predictions if "question_id" in item}
    return {question_id for question_id, item in latest.items() if item.get("error") is None}


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the latest successful prediction for each question."""
    latest = {str(item["question_id"]): item for item in predictions if "question_id" in item}
    successful = [item for item in latest.values() if item.get("error") is None]
    total = len(latest)
    result: dict[str, Any] = {
        "total": total,
        "completed": len(successful),
        "errors": total - len(successful),
        "overall": _accuracy(successful),
        "task_type": _group_accuracy(successful, "task_type"),
        "domain": _group_accuracy(successful, "domain"),
        "sub_category": _group_accuracy(successful, "sub_category"),
    }
    return result


def _accuracy(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    return {
        "accuracy": sum(item.get("prediction") == item.get("answer") for item in items) / count if count else None,
        "n": count,
    }


def _group_accuracy(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key) or "unknown")].append(item)
    return {name: _accuracy(group) for name, group in sorted(groups.items())}
