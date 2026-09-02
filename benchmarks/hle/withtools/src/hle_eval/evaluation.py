from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .harness import (
    HLE_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    AgentResult,
    OpenAICompatibleClient,
    run_agent,
)
from .tools import ToolRunner

DEFAULT_DATASET_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"

ORIGINAL_JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtractedAnswer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "integer"},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}

_MIME_TYPE_FIXES = {
    "671f2c7debcbc34ccb44613b": "image/webp",
    "6736d5dc278519f8b1845091": "image/png",
}


@dataclass
class Sample:
    id: str
    uid: str
    question: str
    target: str
    image: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def prepare_sample(record: dict[str, Any], base_dir: Path | None = None) -> Sample:
    uid = str(record["id"])
    stable = hashlib.md5(uid.encode(), usedforsecurity=False).hexdigest()[:8]
    image = _prepare_image(record.get("image"), uid, base_dir)
    known = {"id", "question", "answer", "image"}
    metadata = {key: value for key, value in record.items() if key not in known}
    metadata["uid"] = uid
    metadata["has_image"] = image is not None
    return Sample(
        id=f"hle_{stable}",
        uid=uid,
        question=str(record["question"]),
        target=str(record["answer"]),
        image=image,
        metadata=metadata,
    )


def load_sample(dataset: dict[str, Any], sample_id: str) -> Sample:
    provider = dataset.get("provider", "jsonl")
    if provider == "jsonl":
        path = Path(_required_string(dataset, "path"))
        records = _read_jsonl(path)
        base_dir = path.parent
    elif provider == "huggingface":
        records = _read_huggingface(dataset)
        base_dir = None
    else:
        raise ValueError("dataset.provider must be jsonl or huggingface")
    for record in records:
        uid = str(record.get("id", ""))
        stable_id = (
            "hle_" + hashlib.md5(uid.encode(), usedforsecurity=False).hexdigest()[:8]
        )
        if sample_id in {uid, stable_id}:
            return prepare_sample(record, base_dir)
    raise ValueError(f"sample ID not found: {sample_id}")


def validate_config(config: dict[str, Any]) -> None:
    max_tool_calls = config.get("max_tool_calls", 120)
    if isinstance(max_tool_calls, bool) or max_tool_calls != 120:
        raise ValueError("max_tool_calls must be 120")

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    if model.get("provider", "openai_compatible") not in {
        "openai",
        "openai_compatible",
    }:
        raise ValueError("model.provider must be openai_compatible")
    _required_string(model, "model")
    _required_string(model, "base_url")
    if "judge" in config and not isinstance(config["judge"], dict):
        raise ValueError("judge must be an object")

    tools = config.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("tools must be an object")
    search = tools.get("web_search") or {}
    if search.get("provider") not in {"tavily", "exa", "google", "http_json"}:
        raise ValueError(
            "web_search.provider must be tavily, exa, google, or http_json"
        )
    if (tools.get("web_fetch") or {}).get("provider", "builtin") != "builtin":
        raise ValueError("web_fetch.provider must be builtin")
    if (tools.get("code_execution") or {}).get("provider", "builtin") != "builtin":
        raise ValueError("code_execution.provider must be builtin")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    if dataset.get("provider", "jsonl") not in {"jsonl", "huggingface"}:
        raise ValueError("dataset.provider must be jsonl or huggingface")
    if dataset.get("provider", "jsonl") == "jsonl":
        _required_string(dataset, "path")

    _required_string(config, "sample_id")


def evaluate(config: dict[str, Any]) -> dict[str, int]:
    validate_config(config)
    sample = load_sample(config["dataset"], config["sample_id"])
    output = Path(config.get("output", "/output/results.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output) if config.get("resume", True) else set()
    solver = OpenAICompatibleClient(config["model"])
    judge = OpenAICompatibleClient(config.get("judge", config["model"]))
    summary = {
        "selected": 1,
        "skipped": 0,
        "success": 0,
        "failed": 0,
        "correct": 0,
    }

    with output.open("a", encoding="utf-8") as stream:
        if sample.id in completed:
            summary["skipped"] = 1
            return summary
        record = _evaluate_sample(sample, solver, judge, config)
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        summary[record["status"]] += 1
        if record.get("score") == 1:
            summary["correct"] += 1
        print(
            json.dumps(
                {
                    "id": sample.id,
                    "status": record["status"],
                    "score": record.get("score"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return summary


def _evaluate_sample(
    sample: Sample,
    solver: OpenAICompatibleClient,
    judge: OpenAICompatibleClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    solver.events.clear()
    judge.events.clear()
    attempts = int(config.get("sample_retries", 1)) + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = run_agent(
                solver,
                ToolRunner(config["tools"]),
                sample.question,
                sample.image,
                int(config.get("max_turns", 100)),
                int(config.get("compaction_tokens", 50_000)),
                int(config.get("max_tool_calls", 120)),
            )
            judgment = judge_answer(
                judge, sample.question, result.answer, sample.target
            )
            return _result_record(
                sample, result, judgment, attempt, solver, judge, config
            )
        except Exception as exc:
            last_error = exc
    return {
        "id": sample.id,
        "uid": sample.uid,
        "status": "failed",
        "score": None,
        "target": sample.target,
        "error": str(last_error),
        "attempts": attempts,
        "model_events": solver.events,
        "judge_events": judge.events,
        "run": _run_record(config),
        "timestamp": _now(),
    }


def judge_answer(
    client: OpenAICompatibleClient,
    question: str,
    answer: str,
    target: str,
) -> dict[str, Any]:
    prompt = ORIGINAL_JUDGE_PROMPT.format(
        question=question,
        response=answer,
        correct_answer=target,
    )
    response = client.complete(
        [{"role": "user", "content": prompt}],
        [],
        response_format=JUDGE_SCHEMA,
    )
    content = response["message"].get("content")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str):
        raise ValueError("judge returned no text content")
    data = _parse_json_object(content)
    required = {"extracted_final_answer", "reasoning", "correct", "confidence"}
    if not required.issubset(data):
        raise ValueError("judge response is missing required fields")
    if data["correct"] not in {"yes", "no"}:
        raise ValueError("judge correct field must be yes or no")
    data["confidence"] = max(0, min(100, int(data["confidence"])))
    data["usage"] = response.get("usage", {})
    return data


def _result_record(
    sample: Sample,
    result: AgentResult,
    judgment: dict[str, Any],
    attempts: int,
    solver: OpenAICompatibleClient,
    judge: OpenAICompatibleClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": sample.id,
        "uid": sample.uid,
        "status": "success",
        "score": 1 if judgment["correct"] == "yes" else 0,
        "answer": result.answer,
        "target": sample.target,
        "judgment": judgment,
        "question": sample.question,
        "metadata": sample.metadata,
        "messages": result.messages,
        "tool_events": result.tool_events,
        "compactions": result.compactions,
        "usage": result.usage,
        "turns": result.turns,
        "attempts": attempts,
        "model_events": solver.events,
        "judge_events": judge.events,
        "run": _run_record(config),
        "timestamp": _now(),
    }


def _run_record(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver_model": config["model"]["model"],
        "judge_model": config.get("judge", config["model"])["model"],
        "tool_providers": {
            name: value.get("provider", "builtin")
            for name, value in config["tools"].items()
        },
        "max_turns": int(config.get("max_turns", 100)),
        "max_tool_calls": int(config.get("max_tool_calls", 120)),
        "compaction_tokens": int(config.get("compaction_tokens", 50_000)),
        "prompts": {
            "react_system": REACT_SYSTEM_PROMPT,
            "hle_system": HLE_SYSTEM_PROMPT,
        },
        "tool_definitions": TOOL_DEFINITIONS,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _read_huggingface(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    repo = str(config.get("repo", "cais/hle"))
    name = str(config.get("config", "default"))
    split = str(config.get("split", "test"))
    revision = str(config.get("revision", DEFAULT_DATASET_REVISION))
    token_env = str(config.get("token_env", "HF_TOKEN"))
    token = os.getenv(token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    endpoint = str(config.get("url", "https://datasets-server.huggingface.co/rows"))
    offset = 0
    page_size = min(100, int(config.get("page_size", 100)))
    while True:
        response = httpx.get(
            endpoint,
            params={
                "dataset": repo,
                "config": name,
                "split": split,
                "revision": revision,
                "offset": offset,
                "length": page_size,
            },
            headers=headers,
            timeout=float(config.get("timeout", 120)),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            hint = (
                f"; set {token_env}" if exc.response.status_code in {401, 403} else ""
            )
            raise RuntimeError(
                f"Hugging Face dataset request failed with HTTP {exc.response.status_code}{hint}"
            ) from exc
        data = response.json()
        rows = data.get("rows", [])
        for entry in rows:
            if isinstance(entry, dict) and isinstance(entry.get("row"), dict):
                yield entry["row"]
        offset += len(rows)
        if not rows or offset >= int(data.get("num_rows_total", offset)):
            break


def _prepare_image(value: Any, uid: str, base_dir: Path | None) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        if isinstance(value.get("src"), str):
            value = value["src"]
        elif isinstance(value.get("bytes"), str):
            mime = str(value.get("mime_type", "image/jpeg"))
            value = f"data:{mime};base64,{value['bytes']}"
        elif isinstance(value.get("path"), str):
            value = value["path"]
    if not isinstance(value, str):
        raise ValueError(f"unsupported image value for record {uid}")
    if value.startswith("data:"):
        correct_mime = _MIME_TYPE_FIXES.get(uid)
        if correct_mime:
            _, _, encoded = value.partition(",")
            return f"data:{correct_mime};base64,{encoded}"
        return value
    if value.startswith(("http://", "https://")):
        return value
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE
        )
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                completed.add(record["id"])
    return completed


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
