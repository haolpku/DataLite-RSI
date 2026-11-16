"""Small OpenAI-compatible JSON client for the configured experiment API.

Credentials are supplied at runtime through environment variables or an
explicitly passed private config file. They are never written into manifests
or logs, and this package never guesses a user-specific credential location.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


def _parse_json_content(content: str) -> dict[str, Any]:
    """Accept strict JSON and the fenced JSON commonly returned by APIs."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("API response JSON must be an object")
    return value


def load_api_config(path: Path | None = None) -> tuple[str, str]:
    base_url = os.environ.get("RSI_API_BASE_URL")
    api_key = os.environ.get("RSI_API_KEY")
    if base_url and api_key:
        return base_url.rstrip("/"), api_key
    if path is None:
        raise ValueError(
            "Set RSI_API_BASE_URL and RSI_API_KEY, or pass an explicit private "
            "--config-path. No default credential file is used."
        )
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'"base_url"\s*:\s*"([^"]+)"\s*,\s*"api_key"\s*:\s*"([^"]+)"',
        text,
    )
    if not match:
        raise ValueError(f"API config block not found in {path}")
    return match.group(1).rstrip("/"), match.group(2)


class OpenAICompatibleJSONClient:
    """Call `/chat/completions` without persisting the API credential."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config_path: Path | None = None,
        timeout_sec: float = 600.0,
    ) -> None:
        if base_url is None or api_key is None:
            configured_url, configured_key = load_api_config(config_path)
            base_url = base_url or configured_url
            api_key = api_key or configured_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_sec = timeout_sec

    def complete_json(
        self,
        *,
        prompt: str,
        system: str = "Return only valid JSON.",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        return _parse_json_content(content)

    def complete_multimodal_json(
        self,
        *,
        content: list[dict[str, Any]],
        system: str = "Return only valid JSON.",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send text plus image/video-compatible content to a vision endpoint."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            result = json.load(response)
        content_value = result["choices"][0]["message"]["content"]
        if isinstance(content_value, list):
            content_value = "".join(
                item.get("text", "") for item in content_value if isinstance(item, dict)
            )
        text = str(content_value).strip()
        try:
            return _parse_json_content(text)
        except (json.JSONDecodeError, ValueError):
            # Some OpenAI-compatible vision gateways ignore response_format
            # and return a bare option letter or a short natural-language
            # answer.  Preserve that content for the caller to normalize
            # instead of silently turning every prediction into an error.
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1]).strip()
            return {"answer": text}
