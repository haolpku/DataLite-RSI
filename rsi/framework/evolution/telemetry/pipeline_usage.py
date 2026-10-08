"""Capture and aggregate LLM token usage from generated pipeline HTTP calls.

The official DataFlow HTTP serving formats successful JSON responses into
plain strings and discards the raw ``usage`` object.  Generated operators therefore cannot
report exact token counts after the fact.  This module instruments ``requests.Session`` in
the pipeline subprocess, before DataFlow creates its sessions, and records only metadata
and usage counters—never prompts or generated content.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


USAGE_PATH_ENV = "DF_PIPELINE_LLM_USAGE_PATH"
EXECUTION_ID_ENV = "DF_PIPELINE_LLM_EXECUTION_ID"
ITERATION_ENV = "DF_PIPELINE_LLM_ITERATION"
ATTEMPT_ENV = "DF_PIPELINE_LLM_ATTEMPT"
BOOTSTRAP_DIR = Path(__file__).resolve().parent / "_bootstrap"

_WRITE_LOCK = threading.Lock()
_INSTALL_LOCK = threading.Lock()
_ORIGINAL_SEND_ATTR = "_dfe_usage_original_send"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_usage(value: Any) -> dict[str, int] | None:
    """Normalize OpenAI/Anthropic-compatible usage without estimating missing values."""
    if not isinstance(value, dict):
        return None
    prompt_details = value.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    completion_details = value.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}

    input_tokens = _as_int(value.get("prompt_tokens", value.get("input_tokens")))
    output_tokens = _as_int(value.get("completion_tokens", value.get("output_tokens")))
    total_tokens = _as_int(value.get("total_tokens"))
    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    cached_input_tokens = _as_int(
        prompt_details.get(
            "cached_tokens",
            value.get("cache_read_input_tokens", value.get("cached_input_tokens")),
        )
    )
    cache_creation_input_tokens = _as_int(value.get("cache_creation_input_tokens"))
    reasoning_tokens = _as_int(
        completion_details.get("reasoning_tokens", value.get("reasoning_tokens"))
    )
    if not any(
        (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            reasoning_tokens,
        )
    ):
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _request_payload(request: Any) -> dict[str, Any]:
    body = getattr(request, "body", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not isinstance(body, str) or not body:
        return {}
    try:
        value = json.loads(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _request_kind(url: str, payload: dict[str, Any]) -> str | None:
    lowered = url.lower()
    if "messages" in payload or "chat/completions" in lowered:
        return "chat_completion"
    if "input" in payload and ("embedding" in lowered or "model" in payload):
        return "embedding"
    return None


def _safe_url(value: str) -> str:
    """Drop query strings/fragments that may contain credentials or signed parameters."""
    try:
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        return ""


def build_usage_event(
    request: Any,
    *,
    response: Any | None,
    elapsed_seconds: float,
    error_type: str | None = None,
) -> dict[str, Any] | None:
    """Build one privacy-preserving event from a prepared request and response."""
    url = str(getattr(request, "url", "") or "")
    payload = _request_payload(request)
    kind = _request_kind(url, payload)
    if kind is None:
        return None

    response_data: dict[str, Any] = {}
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                response_data = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                from rsi.framework.io.serving import aggregate_chat_stream

                response_data = aggregate_chat_stream(response.content) or {}
            except (AttributeError, TypeError, ValueError):
                pass
    usage = normalize_usage(response_data.get("usage"))
    messages = payload.get("messages")
    inputs = payload.get("input")
    if isinstance(messages, list):
        item_count = len(messages)
    elif isinstance(inputs, list):
        item_count = len(inputs)
    else:
        item_count = 1

    status_code = getattr(response, "status_code", None) if response is not None else None
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_id": os.environ.get(EXECUTION_ID_ENV, ""),
        "iteration": os.environ.get(ITERATION_ENV, ""),
        "attempt": _as_int(os.environ.get(ATTEMPT_ENV, "0")),
        "kind": kind,
        "model": str(response_data.get("model") or payload.get("model") or ""),
        "response_id": str(response_data.get("id") or ""),
        "method": str(getattr(request, "method", "POST") or "POST"),
        "url": _safe_url(url),
        "item_count": item_count,
        "status_code": status_code,
        "success": bool(status_code is not None and 200 <= int(status_code) < 300),
        "latency_seconds": round(max(float(elapsed_seconds), 0.0), 6),
        "error_type": error_type,
        "usage_status": "exact" if usage is not None else "missing",
        "usage": usage,
    }


def append_usage_event(path: str | Path, event: dict[str, Any]) -> None:
    """Append one complete JSON line safely across DataFlow worker threads."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _safe_append_usage_event(path: str | Path, event: dict[str, Any] | None) -> None:
    if event is None:
        return
    try:
        append_usage_event(path, event)
    except (OSError, TypeError, ValueError):
        # Accounting must never change the API call or pipeline outcome.
        pass


def _attach_stream_json(response: Any) -> None:
    """Expose an aggregated JSON view while retaining the original SSE bytes.

    Generated pipeline callers may request ``stream=True`` and then call
    ``response.json()``. Usage accounting already reads the completed body;
    without this view that call attempts to parse raw ``data:`` lines as JSON.
    ``iter_lines()`` still sees the original cached SSE content.
    """
    content_type = str(getattr(response, "headers", {}).get("Content-Type", "")).lower()
    if "text/event-stream" not in content_type:
        return
    try:
        from rsi.framework.io.serving import aggregate_chat_stream

        parsed = aggregate_chat_stream(response.content)
    except (AttributeError, TypeError, ValueError):
        return
    if isinstance(parsed, dict):
        response.json = lambda **_kwargs: parsed


def install_requests_usage_tracking() -> bool:
    """Install process-local requests instrumentation when the usage path is configured."""
    destination = os.environ.get(USAGE_PATH_ENV, "").strip()
    if not destination:
        return False
    try:
        import requests
    except ImportError:
        return False

    session_cls = requests.sessions.Session
    with _INSTALL_LOCK:
        if hasattr(session_cls, _ORIGINAL_SEND_ATTR):
            return True
        original_send = session_cls.send
        setattr(session_cls, _ORIGINAL_SEND_ATTR, original_send)

        def tracked_send(self, request, **kwargs):
            started = time.perf_counter()
            try:
                response = original_send(self, request, **kwargs)
            except Exception as exc:
                event = build_usage_event(
                    request,
                    response=None,
                    elapsed_seconds=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
                _safe_append_usage_event(destination, event)
                raise
            _attach_stream_json(response)
            event = build_usage_event(
                request,
                response=response,
                elapsed_seconds=time.perf_counter() - started,
            )
            _safe_append_usage_event(destination, event)
            return response

        session_cls.send = tracked_send
    return True


def _empty_totals() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _add_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key in target:
        target[key] += _as_int(usage.get(key))


def aggregate_pipeline_usage(events_path: str | Path) -> dict[str, Any]:
    """Aggregate all completed calls, including successful retry attempts."""
    path = Path(events_path)
    totals = _empty_totals()
    by_model: dict[str, dict[str, Any]] = {}
    executions: set[str] = set()
    request_count = success_count = failure_count = missing_usage_count = 0

    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                request_count += 1
                if event.get("success"):
                    success_count += 1
                else:
                    failure_count += 1
                execution_id = str(event.get("execution_id") or "")
                if execution_id:
                    executions.add(execution_id)
                usage = event.get("usage")
                if not isinstance(usage, dict):
                    missing_usage_count += 1
                    continue
                _add_usage(totals, usage)
                model = str(event.get("model") or "unknown")
                group = by_model.setdefault(
                    model,
                    {"request_count": 0, "tokens": _empty_totals()},
                )
                group["request_count"] += 1
                _add_usage(group["tokens"], usage)

    return {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(path),
        "execution_ids": sorted(executions),
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "missing_usage_count": missing_usage_count,
        "tokens": totals,
        "by_model": by_model,
    }


def write_pipeline_usage_summary(
    events_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    summary = aggregate_pipeline_usage(events_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return summary


def write_usage_payload(output_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically persist a pre-aggregated model-usage payload."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.setdefault("schema_version", 1)
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return value


def write_combined_iteration_usage(iteration_dir: str | Path) -> dict[str, Any]:
    """Namespace Agent and pipeline usage without inventing an incompatible grand total."""
    root = Path(iteration_dir)

    def read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "agent": read(root / "agent_token_usage.json"),
        "pipeline_llm": read(root / "pipeline_llm_token_usage.json"),
        "review_llm": read(root / "review_llm_token_usage.json"),
        "embedding": read(root / "embedding_token_usage.json"),
    }
    destination = root / "token_usage.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return payload
