#!/usr/bin/env python3
"""Materialize VideoRSI MCQs as Video-MME-aligned ShareGPT SFT records.

The source JSONL remains immutable.  Each output record presents the video,
question, and four choices, while the assistant target is exactly ``A``–``D``.
Choices are deterministically permuted per view so that training cannot learn a
global answer-position prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D")


def as_choices(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        result = {str(key).upper(): str(text).strip() for key, text in value.items()}
    elif isinstance(value, list):
        result = {LETTERS[index]: str(text).strip() for index, text in enumerate(value)}
    else:
        return {}
    return result if set(result) == set(LETTERS) and all(result.values()) else {}


def answer_letter(record: dict[str, Any], choices: dict[str, str]) -> str | None:
    direct = str(record.get("answer_letter", "")).strip().upper()
    if direct in LETTERS:
        return direct
    answer = str(record.get("answer", "")).strip()
    exact = [letter for letter, choice in choices.items() if answer == choice]
    return exact[0] if len(exact) == 1 else None


def permutation(sample_id: str, view: int) -> list[str]:
    ranked = sorted(
        LETTERS,
        key=lambda letter: hashlib.sha256(f"videorsi-shuffle-v1:{sample_id}:{view}:{letter}".encode()).hexdigest(),
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--views", type=int, default=1, choices=range(1, 5))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"videorsi_videomme_{args.views}view.jsonl"
    stats: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source, 1):
            stats["input"] += 1
            record = json.loads(line)
            choices = as_choices(record.get("choices"))
            letter = answer_letter(record, choices)
            video_path = str((record.get("video") or {}).get("uri") or (record.get("video_metadata") or {}).get("path") or "")
            question = str(record.get("question", "")).strip()
            if not choices or not letter or not video_path or not question:
                stats["dropped_invalid"] += 1
                continue
            sample_id = str(record.get("sample_id") or f"line_{line_number}")
            for view in range(args.views):
                old_order = permutation(sample_id, view)
                remapped = {new: old for new, old in zip(LETTERS, old_order)}
                target = next(new for new, old in remapped.items() if old == letter)
                option_lines = "\n".join(f"{new}. {choices[old]}" for new, old in remapped.items())
                item = {
                    "messages": [
                        {"from": "human", "value": f"<video>\n{question}\nOptions:\n{option_lines}\nReply with only A, B, C, or D."},
                        {"from": "gpt", "value": target},
                    ],
                    "videos": [video_path],
                    "sample_id": f"{sample_id}__option_view_{view}",
                    "source_sample_id": sample_id,
                    "pipeline_version": "video_rsi_router_v5_multitask",
                    "answer_target_format": "option_letter_abcd",
                    "option_shuffle": {"version": "v1", "view": view, "old_order": old_order},
                    "source_line": line_number,
                }
                sink.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["written"] += 1

    manifest = {
        "source": str(args.input),
        "output": str(output),
        "views": args.views,
        "statistics": dict(stats),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "semantics": "Video-MME-aligned MCQ supervision with deterministic answer-position balancing.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
