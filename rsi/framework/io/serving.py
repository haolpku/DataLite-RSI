"""Pipeline-owned serving aligned with ``open-dataflow`` 1.0.10.

The response contract generated operators rely on: one aligned ``str`` or
``None`` per logical input, with the pipeline owning transport, concurrency,
retries and request construction. Operators never parse raw responses.

Defaults, headers, request bodies, retry timing and failure classification
match the reference. The upstream ``compat/openai_serving.py`` fixes
are built in rather than installed as runtime patches:

* ``enable_thinking`` is a tri-state request/response policy. An explicit
  boolean is forwarded upstream; ``omit``/``None`` keeps the field out of the
  request entirely. ``true`` returns the ``<think>``/``<answer>`` wrapper,
  while ``false`` and ``omit`` return ``message.content`` only. The reference
  instead wraps whenever ``reasoning_content`` is non-empty, regardless of the
  requested policy.
* ``message.reasoning`` is accepted as an alias for ``reasoning_content``.
* A malformed response returns ``None`` instead of raising ``IndexError`` or
  ``TypeError``, so one bad row cannot abort a pipeline step.
* Buffered streaming SSE is aggregated into an ordinary chat-completion
  response, decoded as UTF-8 so symbols such as ``×`` do not become mojibake.

One deliberate difference: the reference's compat installs the SSE adapter by
patching ``requests.sessions.Session.send`` process-wide. This implementation
decodes inside its own request path, so it never rewrites requests issued by
unrelated code. Generated pipelines observe the same text either way.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from ..core.llm_serving import LLMServingABC
from ..core.runtime_logger import get_logger


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
_WRAPPED_RE = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)


def _is_omitted_enable_thinking(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "omit")


def aggregate_chat_stream(body: bytes | str) -> dict[str, Any] | None:
    """Aggregate a buffered chat-completion SSE stream into response JSON.

    Returns ``None`` when the payload carries no usable chunk, so the caller can
    leave the original response untouched.
    """
    if isinstance(body, bytes):
        # SSE is UTF-8 by specification; requests falls back to ISO-8859-1 when
        # text/event-stream omits a charset.
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body)

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


def format_chat_response(
    response: Mapping[str, Any], *, enable_thinking: bool | str | None
) -> str | None:
    """Apply the tri-state response policy to one chat completion."""
    if _is_omitted_enable_thinking(enable_thinking):
        thinking = False
    elif isinstance(enable_thinking, bool):
        thinking = enable_thinking
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
    if not thinking:
        return content

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if content is None:
        if reasoning:
            content = ""
        else:
            return None
    if _WRAPPED_RE.search(content):
        return content
    return f"<think>{reasoning or ''}</think>\n<answer>{content}</answer>"


class PipelineLLMServing(LLMServingABC):
    """OpenAI-compatible serving constructed and injected by a pipeline.

    ``requests`` is imported on construction rather than at module import, so
    the framework still imports in an environment without it.
    """

    def start_serving(self) -> None:
        self.logger.info("PipelineLLMServing: no local service to start.")
        return

    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_workers: int = 10,
        max_retries: int = 5,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        **configs: Any,
    ) -> None:
        import os

        try:
            import requests
            from requests.adapters import HTTPAdapter
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "PipelineLLMServing requires the requests package"
            ) from exc

        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        self.timeout = (connect_timeout, read_timeout)
        if "timeout" in configs:
            warnings.warn(
                "The `timeout` parameter is deprecated. Please use "
                "`connect_timeout` and `read_timeout` instead.",
                DeprecationWarning,
            )
            self.timeout = (connect_timeout, configs["timeout"])
            configs.pop("timeout")

        # An omitted tri-state policy must not reach the upstream request body.
        # An invalid value is deliberately *not* rejected here: the reference
        # keeps it in configs and fails in format_response, so validation stays
        # at the point where the policy is actually applied.
        if _is_omitted_enable_thinking(configs.get("enable_thinking", "__absent__")):
            configs.pop("enable_thinking", None)

        self.configs = configs
        self.configs.update({"temperature": temperature})

        self.logger = get_logger()

        # Keys stay in the environment; only their variable name is configured.
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = (
                f"Lack of `{key_name_of_api_key}` in environment variables. Please "
                f"set `{key_name_of_api_key}` as your api-key to {api_url} before "
                "using PipelineLLMServing."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self._requests = requests
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers,
            max_retries=0,  # retries are owned by _api_chat_id_retry
            pool_block=True,  # block when the pool is full instead of growing
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        }

    @property
    def enable_thinking(self) -> bool | None:
        """The configured policy, or ``None`` when it is omitted."""
        return self.configs.get("enable_thinking")

    def format_response(self, response: Any, is_embedding: bool = False) -> Any:
        """Format one API response into the operator-visible value."""
        if is_embedding:
            if not isinstance(response, Mapping):
                return []
            data = response.get("data")
            if not isinstance(data, list) or not data:
                return []
            first = data[0]
            return first.get("embedding", []) if isinstance(first, Mapping) else []

        if not isinstance(response, Mapping):
            return None
        return format_chat_response(
            response, enable_thinking=self.configs.get("enable_thinking")
        )

    def _parse_response(self, response: Any) -> Any:
        """Decode the body, aggregating a buffered SSE stream when present."""
        content_type = str(response.headers.get("Content-Type", "")).lower()
        body = getattr(response, "content", b"")
        looks_like_sse = "text/event-stream" in content_type or (
            bool(body)
            and any(
                line.lstrip().startswith(b"data:" if isinstance(body, bytes) else "data:")
                for line in body.splitlines()
            )
        )
        if looks_like_sse:
            aggregated = aggregate_chat_stream(body)
            if aggregated is not None:
                return aggregated
        return response.json()

    def _api_chat_with_id(
        self,
        id: int,
        payload,
        model: str,
        is_embedding: bool = False,
        json_schema: dict = None,
    ):
        requests = self._requests
        start = time.time()
        try:
            if is_embedding:
                payload = {"model": model, "input": payload}
            elif json_schema is None:
                payload = {"model": model, "messages": payload}
            else:
                payload = {
                    "model": model,
                    "messages": payload,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "custom_response",
                            "strict": True,
                            "schema": json_schema,
                        },
                    },
                }

            payload.update(self.configs)
            payload = json.dumps(payload)
            response = self.session.post(
                self.api_url, headers=self.headers, data=payload, timeout=self.timeout
            )
            cost = time.time() - start
            if response.status_code == 200:
                return id, self.format_response(
                    self._parse_response(response), is_embedding
                )
            self.logger.error(
                f"API request failed id={id} status={response.status_code} "
                f"cost={cost:.2f}s body={response.text[:500]}"
            )
            return id, None

        # Connect-stage timeout: the server is unreachable. Raise so the failure
        # is uniform across platforms rather than silently becoming a null row.
        except requests.exceptions.ConnectTimeout as e:
            cost = time.time() - start
            self.logger.error(f"API connect timeout (id={id}) cost={cost:.2f}s: {e}")
            raise RuntimeError(
                f"Cannot connect to LLM server (connect timeout): {e}"
            ) from e

        # Read timeout: reachable but still working. Warn and yield no result.
        except requests.exceptions.ReadTimeout as e:
            cost = time.time() - start
            warnings.warn(
                f"API read timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning
            )
            return id, None

        except requests.exceptions.Timeout as e:
            cost = time.time() - start
            warnings.warn(f"API timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning)
            return id, None

        # requests/urllib3 wrap several distinct faults as ConnectionError, so
        # classify by message before deciding whether this is fatal.
        except requests.exceptions.ConnectionError as e:
            cost = time.time() - start
            msg = str(e).lower()
            if "read timed out" in msg:
                warnings.warn(
                    f"API read timeout (id={id}) cost={cost:.2f}s: {e}", RuntimeWarning
                )
                return id, None
            if "connect timeout" in msg or ("timed out" in msg and "connect" in msg):
                self.logger.error(f"API connect timeout (id={id}) cost={cost:.2f}s: {e}")
                raise RuntimeError(
                    f"Cannot connect to LLM server (connect timeout): {e}"
                ) from e
            self.logger.error(f"API connection error (id={id}) cost={cost:.2f}s: {e}")
            raise RuntimeError(f"Cannot connect to LLM server: {e}") from e

        except Exception as e:
            cost = time.time() - start
            self.logger.exception(f"API request error (id = {id}) cost={cost:.2f}s: {e}")
            return id, None

    def _api_chat_id_retry(
        self, id, payload, model, is_embedding: bool = False, json_schema: dict = None
    ):
        for i in range(self.max_retries):
            id, response = self._api_chat_with_id(
                id, payload, model, is_embedding, json_schema
            )
            if response is not None:
                return id, response
            time.sleep(2**i)
        return id, None

    def _run_threadpool(self, task_args_list: list[dict], desc: str) -> list:
        """Run every task concurrently and refill results by input id."""
        from tqdm import tqdm

        responses = [None] * len(task_args_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._api_chat_id_retry, **task_args)
                for task_args in task_args_list
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    response = future.result()  # (id, response)
                    responses[response[0]] = response[1]
                except Exception:
                    # Workers already handle their own errors; this is a backstop
                    # so one crashed worker cannot lose the whole batch.
                    self.logger.exception("Worker crashed unexpectedly in threadpool")
        return responses

    def generate_from_input(
        self,
        user_inputs: Sequence[str],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        json_schema: dict = None,
    ) -> list[str | None]:
        task_args_list = [
            dict(
                id=idx,
                payload=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                model=self.model_name,
                json_schema=json_schema,
            )
            for idx, question in enumerate(user_inputs)
        ]
        return self._run_threadpool(
            task_args_list, desc="Generating responses from prompts......"
        )

    def generate_from_conversations(
        self, conversations: Sequence[Sequence[dict]]
    ) -> list[str | None]:
        task_args_list = [
            dict(id=idx, payload=dialogue, model=self.model_name)
            for idx, dialogue in enumerate(conversations)
        ]
        return self._run_threadpool(
            task_args_list, desc="Generating responses from conversations......"
        )

    def generate_embedding_from_input(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        task_args_list = [
            dict(id=idx, payload=txt, model=self.model_name, is_embedding=True)
            for idx, txt in enumerate(texts)
        ]
        return self._run_threadpool(task_args_list, desc="Generating embedding......")

    def cleanup(self) -> None:
        self.logger.info("Cleaning up resources in PipelineLLMServing")
        try:
            if getattr(self, "session", None):
                self.session.close()
        except Exception:
            self.logger.exception("Failed to close requests session")
