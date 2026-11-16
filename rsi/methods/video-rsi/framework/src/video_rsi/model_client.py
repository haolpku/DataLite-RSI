"""Text-only structured JSON client for RSI operators."""

from __future__ import annotations

import json
from typing import Any, Protocol


class TextJSONModelClient(Protocol):
    model: str

    def complete_json(
        self, *, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]: ...


class ModelOutputTruncatedError(ValueError):
    pass


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class VLLMTextJSONClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        seed: int = 0,
        timeout_sec: float = 600.0,
        disable_thinking: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - server dependency
            raise RuntimeError("Install openai or use a custom TextJSONModelClient") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.disable_thinking = disable_thinking

    def complete_json(
        self, *, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"structured_outputs": {"json": schema}}
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned empty content")
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise ModelOutputTruncatedError(
                f"model output hit max_tokens={max_tokens}; content_chars={len(content)}"
            )
        try:
            return parse_json_object(content)
        except json.JSONDecodeError as exc:
            tail = content[-240:].replace("\n", "\\n")
            raise ValueError(
                f"invalid JSON response; finish_reason={finish_reason!r}; "
                f"content_chars={len(content)}; tail={tail!r}"
            ) from exc

