"""AgentABC：非-SDK 辅助 Agent 的统一基类。

封装“拼 prompt → 调 LLM → 解析（可重试）”的样板。ReviewAgent 继承它；
PipelineAgent 不继承（它走自主编码 AgentRuntime）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dataflow_evolver.llm.serving import LLMServingABC
from dataflow_evolver.utils.logging import get_logger
from dataflow_evolver.utils.parsing import extract_json


class AgentABC(ABC):
    """带 LLM 句柄、重试与 JSON 解析的辅助 Agent 基类。"""

    name: str = "agent"

    def __init__(self, serving: LLMServingABC, max_retries: int = 2) -> None:
        self.serving = serving
        self.max_retries = max_retries
        self.logger = get_logger(f"agents.{self.name}")
        # 最近一次 LLM 的原始文本响应。run_json 只返回抽取后的 dict，原文此前仅在解析失败时
        # 截 200 字打进日志；留在这里供调用方落盘存证（判断打分是否可信、离线复算）。
        self.last_raw_response: str | None = None

    @abstractmethod
    def build_prompt(self, *args: Any, **kwargs: Any) -> str:
        """子类据输入拼出发给 LLM 的 prompt。"""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: str, *args: Any, **kwargs: Any) -> Any:
        """子类把 LLM 原始文本解析为结构化产物（如某个 dataclass）。"""
        raise NotImplementedError

    def run_json(self, prompt: str) -> Any:
        """调用 LLM 并要求 JSON；调用异常或解析失败均重试，最终失败返回 None。

        关键：把 `serving.generate_one` 的**任何异常**（网络/超时/鉴权等）接住并降级为
        一次失败尝试，绝不抛到主循环——与 executor/SDK 的"失败不拖垮编排器"
        原则保持一致（见 code review H1）。
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.serving.generate_one(prompt)
            except Exception as exc:  # noqa: BLE001 —— 故意兜底所有后端异常
                last_err = exc
                self.last_raw_response = None
                self.logger.warning(
                    "[%s] LLM 调用异常（第 %d/%d 次），重试中：%s",
                    self.name, attempt + 1, self.max_retries + 1, exc,
                )
                continue
            self.last_raw_response = raw
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            last_err = ValueError("无法从 LLM 输出中解析出 JSON")
            self.logger.warning(
                "[%s] JSON 解析失败（第 %d/%d 次），重试中。原始输出片段：%s",
                self.name, attempt + 1, self.max_retries + 1, raw[:200],
            )
        self.logger.error("[%s] 多次重试后仍失败：%s", self.name, last_err)
        return None
