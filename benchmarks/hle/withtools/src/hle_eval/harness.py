from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import httpx

HLE_SYSTEM_PROMPT = """Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

HLE_PROMPT = """You are solving a Humanity's Last Exam question.

Work carefully and use the available tools when they improve reliability:
search for current or obscure facts, fetch primary sources before trusting a
claim, and use code_execution for arithmetic, symbolic work, or data checks.
Keep track of evidence and reconcile conflicting sources. Before submitting,
format the response exactly as required by the HLE system prompt. Pass the
entire Explanation, Answer, and Confidence response to the submit tool.
"""

ASSISTANT_PROMPT = """
You are a helpful assistant attempting to submit the best possible answer. You have several tools available to help with finding the answer. You will see the result of tool calls right after sending the message. Prioritize parallel tool calls: when operations are independent, run them in one response — e.g. reading several files or running several searches at once — rather than one at a time. Only sequence calls when one depends on another's result. Do some reasoning before your actions, describing what tool calls you are going to use and how they fit into your plan.
"""

SUBMIT_PROMPT = """
When you have completed the task and have an answer, call the submit() tool to report it.
"""

REACT_SYSTEM_PROMPT = HLE_PROMPT + "\n\n" + ASSISTANT_PROMPT + "\n" + SUBMIT_PROMPT

CONTINUE_PROMPT = """
Please proceed to the next step using your best judgement. If you believe you have completed the task, please call the `submit()` tool with your final answer.
"""

TOOL_BUDGET_PROMPTS = {
    100: (
        "You have used 100 of 120 available research tool calls. Begin "
        "converging now: review the evidence already collected, resolve only "
        "the most important remaining uncertainty, and prepare your final answer."
    ),
    110: (
        "Urgent: you have used 110 of 120 available research tool calls. Stop "
        "broad exploration. Use the remaining calls only for decisive checks, "
        "then synthesize the evidence and submit your answer promptly."
    ),
    115: (
        "Final warning: you have used 115 of 120 available research tool calls. "
        "Only five calls remain. Do not start new lines of investigation; make "
        "at most indispensable verification calls and submit your best answer now."
    ),
    120: (
        "The research tool budget is exhausted. No further search, fetch, or "
        "code execution is available. Use the evidence already gathered and "
        "call submit with your best final answer now."
    ),
}

CODE_DESCRIPTION = """Use the python function to execute Python code.

The Python tool executes single-run Python scripts. Important notes:
1. Each execution is independent - no state is preserved between runs
2. You must explicitly use print() statements to see any output
3. Simply writing expressions (like in notebooks) will not display results
4. The script cannot accept interactive input during execution
5. Return statements alone won't produce visible output
6. All variables and imports are cleared between executions
7. Standard output (via print()) is the only way to see results"""


def _function(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOL_DEFINITIONS = [
    _function(
        "web_search",
        "Use the web_search tool to perform keyword searches of the web.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    _function(
        "web_fetch",
        "Fetch one HTTP(S) URL.",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute URL using the HTTP or HTTPS scheme.",
                },
                "max_length": {
                    "description": (
                        "Maximum characters returned for this call. Defaults to\n"
                        "the factory's ``max_chars`` value."
                    ),
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                },
                "start_index": {
                    "type": "integer",
                    "description": "Character offset for continuing a truncated result.",
                    "default": 0,
                },
                "raw": {
                    "type": "boolean",
                    "description": "Return the original response instead of simplifying HTML.",
                    "default": False,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
    _function(
        "code_execution",
        CODE_DESCRIPTION,
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ),
    _function(
        "submit",
        "Submit an answer for evaluation.",
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Submitted answer"}
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    ),
]


SUMMARY_PROMPT = """You have been working on the task described above but have not yet completed it. Write a continuation summary that will allow you (or another instance of yourself) to resume work efficiently in a future context window where the conversation history will be replaced with this summary. Your summary should be structured, concise, and actionable. Preserve the question, candidate answer, source URLs, key extracted facts, calculations, and unresolved checks."""


@dataclass
class AgentResult:
    answer: str
    messages: list[dict[str, Any]]
    turns: int
    usage: dict[str, int] = field(default_factory=dict)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    compactions: list[dict[str, Any]] = field(default_factory=list)


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any]) -> None:
        provider = config.get("provider", "openai_compatible")
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError("model.provider must be openai_compatible")
        self.model = _required_string(config, "model")
        self.base_url = _required_string(config, "base_url").rstrip("/")
        self.endpoint = config.get("endpoint", f"{self.base_url}/chat/completions")
        self.timeout = float(config.get("timeout", 600))
        self.retries = int(config.get("retries", 3))
        self.parameters = dict(config.get("parameters", {}))
        self.headers = {str(k): str(v) for k, v in config.get("headers", {}).items()}
        self.events: list[dict[str, Any]] = []
        key = config.get("api_key")
        key_env = config.get("api_key_env")
        if key is not None and (not isinstance(key, str) or not key):
            raise ValueError("api_key must be a non-empty string")
        if key is None and key_env:
            key = os.getenv(str(key_env))
            if not key:
                raise ValueError(f"environment variable {key_env!r} is not set")
        if key:
            self.headers.setdefault("Authorization", f"Bearer {key}")

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        protected = {"model", "messages", "tools", "tool_choice", "response_format"}
        collision = protected.intersection(self.parameters)
        if collision:
            raise ValueError(
                f"model.parameters cannot override: {', '.join(sorted(collision))}"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.parameters,
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            try:
                response = httpx.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise ValueError("model response message must be an object")
                self.events.append(
                    {
                        "input": deepcopy(messages),
                        "tools": deepcopy(tools),
                        "response_format": deepcopy(response_format),
                        "output": data,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                return {
                    "message": _assistant_message(message),
                    "usage": data.get("usage") or {},
                    "model": data.get("model", self.model),
                }
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                self.events.append(
                    {
                        "input": deepcopy(messages),
                        "tools": deepcopy(tools),
                        "response_format": deepcopy(response_format),
                        "error": str(exc),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if attempt >= self.retries or not retryable:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"model request failed: {last_error}") from last_error


def run_agent(
    model: Any,
    tool_runner: Any,
    question: str,
    image: str | None,
    max_turns: int,
    compaction_tokens: int,
    max_tool_calls: int = 120,
) -> AgentResult:
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": image}})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "system", "content": HLE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    usage: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    compactions: list[dict[str, Any]] = []
    tool_call_count = 0
    notified_thresholds: set[int] = set()

    for turn in range(1, max_turns + 1):
        if _estimated_tokens(messages) >= compaction_tokens:
            before = messages
            messages = _compact(model, messages)
            compactions.append({"before": before, "after": messages})

        available_tools = (
            TOOL_DEFINITIONS
            if tool_call_count < max_tool_calls
            else [TOOL_DEFINITIONS[-1]]
        )
        response = model.complete(list(messages), available_tools)
        _add_usage(usage, response.get("usage", {}))
        assistant = response["message"]
        if not isinstance(assistant, dict):
            raise RuntimeError("model client returned an invalid message")
        assistant.setdefault("role", "assistant")
        messages.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": CONTINUE_PROMPT})
            continue

        submitted: str | None = None
        crossed_thresholds: list[int] = []
        for call in calls:
            call_id = str(call.get("id", ""))
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                if name == "submit":
                    answer = arguments.get("answer")
                    if not isinstance(answer, str):
                        raise ValueError("submit.answer must be a string")
                    output = answer
                    submitted = answer
                else:
                    if tool_call_count >= max_tool_calls:
                        raise RuntimeError(
                            f"research tool call limit reached ({max_tool_calls})"
                        )
                    tool_call_count += 1
                    output = tool_runner.execute(name, arguments)
                    for threshold in TOOL_BUDGET_PROMPTS:
                        if (
                            tool_call_count >= threshold
                            and threshold not in notified_thresholds
                        ):
                            notified_thresholds.add(threshold)
                            crossed_thresholds.append(threshold)
                event = {
                    "id": call_id,
                    "function": name,
                    "arguments": arguments,
                    "result": output,
                }
            except Exception as exc:
                output = f"Error: {exc}"
                event = {"id": call_id, "function": name, "error": str(exc)}
            tool_events.append(event)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": str(output)}
            )
        for threshold in crossed_thresholds:
            messages.append(
                {"role": "user", "content": TOOL_BUDGET_PROMPTS[threshold]}
            )
        if submitted is not None:
            return AgentResult(
                answer=submitted,
                messages=messages,
                turns=turn,
                usage=usage,
                tool_events=tool_events,
                compactions=compactions,
            )
    raise RuntimeError(f"agent reached turn limit ({max_turns}) without submit")


def _compact(model: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = model.complete(
        messages + [{"role": "user", "content": SUMMARY_PROMPT}], []
    )
    summary = response["message"].get("content") or ""
    summary_message = {
        "role": "user",
        "content": (
            "[CONTEXT COMPACTION SUMMARY]\n\n"
            "The following is a summary of work completed on this task so far:\n\n"
            f"<summary>\n{summary}\n</summary>\n\n"
            "Please continue working on this task from where you left off."
        ),
    }
    return messages[:3] + [summary_message]


def _estimated_tokens(messages: list[dict[str, Any]]) -> int:
    def without_images(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                return {"type": "image_url", "image_url": "<image>"}
            return {key: without_images(item) for key, item in value.items()}
        if isinstance(value, list):
            return [without_images(item) for item in value]
        return value

    return max(1, len(json.dumps(without_images(messages), ensure_ascii=False)) // 4)


def _add_usage(total: dict[str, int], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    for key in ("tool_calls", "reasoning_content", "refusal"):
        if message.get(key) is not None:
            result[key] = message[key]
    return result


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value
