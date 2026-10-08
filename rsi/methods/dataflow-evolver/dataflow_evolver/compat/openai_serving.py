"""Runtime compatibility for DataFlow's native API LLM serving.

Generated pipelines use :class:`APILLMServing_request` directly. This module is
installed by the pipeline subprocess bootstrap and only adapts three wire-format
differences that DataFlow 1.0.x does not handle itself:

* aggregate OpenAI-compatible streaming SSE into a normal chat-completion JSON
  response;
* accept ``message.reasoning`` as an alias for ``message.reasoning_content``;
  and
* treat ``enable_thinking`` as an optional tri-state request/response policy.

DataFlow remains responsible for request construction, retries, concurrency,
and input alignment. Generated pipelines still construct and call only its
native serving class.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import requests
from dataflow.serving.api_llm_serving_request import APILLMServing_request


_SEND_MARKER = "_dataflow_evolver_streaming_reasoning_compat"
_FORMAT_MARKER = "_dataflow_evolver_reasoning_response_compat"
_INIT_MARKER = "_dataflow_evolver_optional_thinking_compat"


def install_api_llm_reasoning_compat() -> None:
    """Install the transparent runtime adapters once per interpreter."""

    _install_optional_enable_thinking_adapter()
    _install_streaming_response_adapter()
    _install_reasoning_response_adapter()


def _is_omitted_enable_thinking(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() == "omit"
    )


def _install_optional_enable_thinking_adapter() -> None:
    current_init = APILLMServing_request.__init__
    if getattr(current_init, _INIT_MARKER, False):
        return

    def init_without_omitted_thinking(self, *args, **kwargs):
        value = kwargs.get("enable_thinking")
        if _is_omitted_enable_thinking(value):
            kwargs.pop("enable_thinking", None)
        return current_init(self, *args, **kwargs)

    setattr(init_without_omitted_thinking, _INIT_MARKER, True)
    setattr(init_without_omitted_thinking, "_original_init", current_init)
    APILLMServing_request.__init__ = init_without_omitted_thinking


def normalize_streaming_chat_response(
    request: Any,
    response: requests.Response,
) -> requests.Response:
    """Turn a buffered streaming chat response into ordinary response JSON."""

    request_payload = _request_json(request)
    if not request_payload or request_payload.get("stream") is not True:
        return response
    if not _is_chat_completions_request(request, request_payload):
        return response
    if getattr(response, "status_code", 0) >= 400:
        return response

    body = getattr(response, "content", b"")
    if not body:
        return response
    if isinstance(body, bytes):
        # SSE is UTF-8 by specification.  ``requests`` otherwise falls back to
        # ISO-8859-1 when ``text/event-stream`` omits a charset, corrupting
        # symbols such as ``×`` into mojibake such as ``Ã``.
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body)
    if not _looks_like_sse(response, text):
        return response

    payload = _aggregate_sse(text)
    if payload is None:
        return response

    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response._content = encoded
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Content-Length"] = str(len(encoded))
    return response


def _install_streaming_response_adapter() -> None:
    current_send = requests.sessions.Session.send
    if getattr(current_send, _SEND_MARKER, False):
        return

    def send_with_streaming_reasoning(session, request, **kwargs):
        response = current_send(session, request, **kwargs)
        try:
            return normalize_streaming_chat_response(request, response)
        except Exception:
            # Leave the original response intact. DataFlow will surface/retry a
            # malformed upstream response through its existing error handling.
            return response

    setattr(send_with_streaming_reasoning, _SEND_MARKER, True)
    setattr(send_with_streaming_reasoning, "_original_send", current_send)
    requests.sessions.Session.send = send_with_streaming_reasoning


def _install_reasoning_response_adapter() -> None:
    current_format = APILLMServing_request.format_response
    if getattr(current_format, _FORMAT_MARKER, False):
        return

    def format_with_reasoning_policy(self, response, is_embedding=False):
        if is_embedding:
            return current_format(self, response, is_embedding=True)
        if not isinstance(response, Mapping):
            return current_format(self, response, is_embedding=False)

        raw_enable_thinking = getattr(self, "configs", {}).get(
            "enable_thinking", None
        )
        if _is_omitted_enable_thinking(raw_enable_thinking):
            enable_thinking = False
        elif isinstance(raw_enable_thinking, bool):
            enable_thinking = raw_enable_thinking
        else:
            raise TypeError("enable_thinking must be a bool, omit, or None")

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            return None
        message = first_choice.get("message") or {}
        if not isinstance(message, Mapping):
            return None

        content = message.get("content")
        if not enable_thinking:
            return content

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if content is None:
            if reasoning:
                content = ""
            else:
                return None
        if re.search(
            r"<think>.*?</think>.*?<answer>.*?</answer>",
            content,
            re.DOTALL,
        ):
            return content
        return f"<think>{reasoning or ''}</think>\n<answer>{content}</answer>"

    setattr(format_with_reasoning_policy, _FORMAT_MARKER, True)
    setattr(
        format_with_reasoning_policy,
        "_original_format_response",
        current_format,
    )
    APILLMServing_request.format_response = format_with_reasoning_policy


def _request_json(request: Any) -> dict[str, Any] | None:
    body = getattr(request, "body", None)
    if body is None:
        return None
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not isinstance(body, str):
        return None
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_chat_completions_request(request: Any, payload: Mapping[str, Any]) -> bool:
    url = str(getattr(request, "url", "")).split("?", 1)[0].rstrip("/")
    return url.endswith("/chat/completions") and isinstance(
        payload.get("messages"), list
    )


def _looks_like_sse(response: requests.Response, text: str) -> bool:
    content_type = str(
        getattr(response, "headers", {}).get("Content-Type", "")
    ).lower()
    return "text/event-stream" in content_type or any(
        line.lstrip().startswith("data:") for line in text.splitlines()
    )


def _aggregate_sse(text: str) -> dict[str, Any] | None:
    chunks: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except (TypeError, ValueError):
            continue
        if isinstance(chunk, dict):
            chunks.append(chunk)
    if not chunks:
        return None

    response: dict[str, Any] = {"object": "chat.completion"}
    choices: dict[int, dict[str, Any]] = {}
    usage = None
    for chunk in chunks:
        for key in ("id", "model", "created", "system_fingerprint"):
            if chunk.get(key) is not None:
                response[key] = chunk[key]
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        for position, raw_choice in enumerate(chunk.get("choices") or []):
            if not isinstance(raw_choice, Mapping):
                continue
            index = raw_choice.get("index", position)
            if not isinstance(index, int):
                index = position
            target = choices.setdefault(
                index,
                {
                    "index": index,
                    "finish_reason": None,
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "",
                        "reasoning": "",
                        "content": "",
                    },
                },
            )
            if raw_choice.get("finish_reason") is not None:
                target["finish_reason"] = raw_choice["finish_reason"]
            delta = raw_choice.get("delta") or raw_choice.get("message") or {}
            if not isinstance(delta, Mapping):
                continue
            message = target["message"]
            if delta.get("role"):
                message["role"] = delta["role"]
            for field in ("reasoning_content", "reasoning", "content"):
                value = delta.get(field)
                if value is not None:
                    message[field] += str(value)

    if not choices:
        return None
    for choice in choices.values():
        message = choice["message"]
        if not message["reasoning_content"]:
            message.pop("reasoning_content")
        if not message["reasoning"]:
            message.pop("reasoning")
    response["choices"] = [choices[index] for index in sorted(choices)]
    if usage is not None:
        response["usage"] = usage
    return response
