"""非-SDK 模型后端：仅供辅助 Agent（Profiler/Review/Diagnostic）批量文本生成使用。

提供统一的 `LLMServingABC.generate(prompts) -> list[str]` 抽象，按 YAML 配置在
OpenAI-compatible API 与本地 vLLM 之间切换。
- api 后端惰性依赖 openai 客户端；
- vllm 后端惰性依赖 vllm + transformers，直接加载本地/HF 模型。

注意：
- 本模块与 DataFlow 运行时完全解耦。生成的 pipeline（DataFlow 运行时）用的是
  DataFlow 自带的 `dataflow.serving.APILLMServing_request`，不走这里；
- PipelineAgent 也不走这里，它使用配置的 AgentRuntime（Claude 或 OpenCode）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any

from dataflow_evolver.telemetry.pipeline_usage import append_usage_event, normalize_usage
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.serving")


class LLMServingABC(ABC):
    """统一的批量文本生成接口。"""

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """对一批 prompt 返回一批补全文本，顺序与输入一一对应。"""
        raise NotImplementedError

    def generate_one(self, prompt: str, **kwargs: Any) -> str:
        """单条便捷封装。"""
        return self.generate([prompt], **kwargs)[0]

    @contextmanager
    def capture_usage(
        self,
        path: str | Path,
        *,
        phase: str,
        kind: str = "review_llm",
    ):
        """Route usage-only events for this logical operation to one JSONL file."""
        previous = getattr(self, "_usage_capture", None)
        self._usage_capture = (str(path), phase, kind)
        try:
            yield
        finally:
            self._usage_capture = previous

    def _record_usage(
        self,
        *,
        model: str,
        raw_usage: Any,
        success: bool,
        latency_seconds: float,
        item_count: int = 1,
        error_type: str | None = None,
    ) -> None:
        capture = getattr(self, "_usage_capture", None)
        if not capture:
            return
        path, phase, kind = capture
        usage_value = _model_dump(raw_usage)
        usage = normalize_usage(usage_value)
        event = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "kind": kind,
            "model": model,
            "item_count": item_count,
            "success": success,
            "latency_seconds": round(max(latency_seconds, 0.0), 6),
            "error_type": error_type,
            "usage_status": "exact" if usage is not None else "missing",
            "usage": usage,
        }
        try:
            append_usage_event(path, event)
        except (OSError, TypeError, ValueError):
            logger.warning("模型 token usage 落盘失败（不影响调用）：%s", path)


class APILLMServing(LLMServingABC):
    """OpenAI-compatible HTTP API 后端。用线程池并发跑一批 prompt。"""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: str,
        max_workers: int = 10,
        temperature: float = 0.7,
        request_timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI  # 惰性导入，避免无谓依赖

        self.model_name = model_name
        # Environment placeholders are expanded as strings by Config. Normalize
        # numeric values at this boundary before passing them to OpenAI/httpx.
        self.max_workers = int(max_workers)
        self.temperature = float(temperature)
        normalized_timeout = float(request_timeout)
        normalized_retries = int(max_retries)
        base_url = _to_base_url(api_url)
        # timeout / max_retries 交给 openai 客户端：网络抖动在 HTTP 层自动退避重试（见 review H1）
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=normalized_timeout,
            max_retries=normalized_retries,
        )
        logger.info("APILLMServing 就绪：model=%s base_url=%s", model_name, base_url)

    def _one(self, prompt: str, **kwargs: Any) -> str:
        logger.debug(
            "API 请求：model=%s, prompt_len=%d, temperature=%.2f",
            self.model_name, len(prompt), kwargs.get("temperature", self.temperature),
        )
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
            )
        except Exception as exc:
            self._record_usage(
                model=self.model_name,
                raw_usage=None,
                success=False,
                latency_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
            )
            raise
        # Some OpenAI-compatible relays return the response body as a plain
        # string instead of an SDK ChatCompletion object. Treat that body as
        # the completion text so review calls do not fail with
        # ``'str' object has no attribute 'choices'``.
        if isinstance(resp, str):
            result = resp
            response_model = self.model_name
            response_usage = None
        else:
            response_model = str(getattr(resp, "model", None) or self.model_name)
            response_usage = getattr(resp, "usage", None)
            result = resp.choices[0].message.content or ""
        self._record_usage(
            model=response_model,
            raw_usage=response_usage,
            success=True,
            latency_seconds=time.perf_counter() - started,
        )
        logger.debug("API 响应：len=%d, preview=%.200s", len(result), result)
        return result

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self._one(prompts[0], **kwargs)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda p: self._one(p, **kwargs), prompts))


class VLLMServing(LLMServingABC):
    """本地 vLLM 后端，直接用 vllm + transformers 加载模型。

    prompts 会经 tokenizer 的 chat template 组装成对话再交给 vLLM 生成，行为与
    DataFlow 的 LocalModelLLMServing_vllm 一致，但实现自包含。
    """

    def __init__(
        self,
        hf_model_name_or_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 1024,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        system_prompt: str = "You are a helpful assistant",
    ) -> None:
        import os

        try:
            from vllm import LLM, SamplingParams  # 惰性导入（重依赖）
        except ImportError as e:  # pragma: no cover - 取决于运行环境
            raise ImportError("使用 vllm 后端需先安装 vllm：`pip install vllm`") from e
        from transformers import AutoTokenizer

        # vLLM 要求多进程用 spawn；见 https://docs.vllm.ai/en/latest/design/multiprocessing.html
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.system_prompt = system_prompt
        self.model_name = hf_model_name_or_path
        self._sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        self._llm = LLM(
            model=hf_model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)
        logger.info("VLLMServing 就绪：model=%s tp=%d", hf_model_name_or_path, tensor_parallel_size)

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        if not prompts:
            return []
        conversations = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]
        templated = self._tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        started = time.perf_counter()
        try:
            outputs = self._llm.generate(templated, self._sampling)
        except Exception as exc:
            self._record_usage(
                model=self.model_name,
                raw_usage=None,
                success=False,
                latency_seconds=time.perf_counter() - started,
                item_count=len(prompts),
                error_type=type(exc).__name__,
            )
            raise
        input_tokens = sum(len(getattr(output, "prompt_token_ids", None) or []) for output in outputs)
        output_tokens = sum(
            len(getattr(completion, "token_ids", None) or [])
            for output in outputs
            for completion in list(getattr(output, "outputs", None) or [])[:1]
        )
        self._record_usage(
            model=self.model_name,
            raw_usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            success=True,
            latency_seconds=time.perf_counter() - started,
            item_count=len(prompts),
        )
        return [o.outputs[0].text for o in outputs]


def build_serving(cfg: Config) -> LLMServingABC:
    """按 `llm` 配置段构造后端。`cfg` 是 `llm` 子配置（含 backend / api / vllm）。"""
    backend = cfg.get("backend", "api")
    if backend == "api":
        api = cfg.api
        # `api_key` 直接是密钥明文；想避免入库可在 YAML 写 ${ENV_VAR}，加载时已展开（见 utils/config.py）。
        api_key = api.get("api_key", "")
        if not api_key:
            logger.warning("未配置 llm.api.api_key（可填明文或 ${ENV_VAR}）；API 调用很可能 401。")
        return APILLMServing(
            api_url=api.api_url,
            model_name=api.model_name,
            api_key=api_key,
            max_workers=api.get("max_workers", 10),
            request_timeout=api.get("request_timeout", 120.0),
            max_retries=api.get("max_retries", 2),
        )
    if backend == "vllm":
        vllm = cfg.vllm
        return VLLMServing(
            hf_model_name_or_path=vllm.hf_model_name_or_path,
            tensor_parallel_size=vllm.get("tensor_parallel_size", 1),
            temperature=vllm.get("temperature", 0.7),
            top_p=vllm.get("top_p", 0.9),
            top_k=vllm.get("top_k", 40),
            max_tokens=vllm.get("max_tokens", 1024),
            repetition_penalty=vllm.get("repetition_penalty", 1.0),
            seed=vllm.get("seed", None),
            max_model_len=vllm.get("max_model_len", None),
            gpu_memory_utilization=vllm.get("gpu_memory_utilization", 0.9),
        )
    raise ValueError(f"未知的 llm.backend: {backend!r}（应为 'api' 或 'vllm'）")


def _to_base_url(api_url: str) -> str:
    """把完整 chat/completions URL 归一化为 openai 客户端要的 base_url。"""
    suffix = "/chat/completions"
    if api_url.endswith(suffix):
        return api_url[: -len(suffix)]
    return api_url.rstrip("/")


def _model_dump(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        return result if isinstance(result, dict) else None
    return None
