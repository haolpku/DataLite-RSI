"""Bind the parity cases to one implementation.

Each adapter exposes the same ``api`` bundle, so ``cases.py`` never imports
either implementation directly:

* :func:`reference_api` binds the real ``open-dataflow==1.0.10`` package plus
  the DataFlow-Evolver ``compat`` fixes read from the external read-only
  repository. Together those are the effective baseline.
* :func:`framework_api` binds ``rsi.framework``.

The serving probes need HTTP responses without a network, so both adapters
build their request capture and failure injection on the implementation's own
session, never on a shared global patch.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterator


KEY_ENV = "DF_PARITY_API_KEY"
KEY_VALUE = "parity-test-key"
API_URL = "http://parity.invalid/v1/chat/completions"
MODEL = "parity-model"


# --------------------------------------------------------------------------
# shared fake transport
# --------------------------------------------------------------------------

def _json_response(requests_module, payload: dict[str, Any], status: int = 200):
    response = requests_module.Response()
    response.status_code = status
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def _sse_response(requests_module, chunks: list[dict[str, Any]]):
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
    lines.append("data: [DONE]")
    response = requests_module.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "text/event-stream"
    response._content = ("\n\n".join(lines) + "\n\n").encode("utf-8")
    return response


DEFAULT_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "C", "reasoning_content": "R"}}]
}


def _install_capture(requests_module, recorded: list[dict[str, Any]]):
    """Record every request body and reply with a fixed chat completion."""
    def fake_post(self, url, headers=None, data=None, timeout=None, **kwargs):
        recorded.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "body": json.loads(data),
                "timeout": list(timeout) if timeout else None,
            }
        )
        return _json_response(requests_module, DEFAULT_REPLY)

    return fake_post


def _failure_post(requests_module, mode: str, counter: dict[str, int]):
    """Return a post() that reproduces one transport failure mode."""
    exceptions = requests_module.exceptions

    def fake_post(self, url, headers=None, data=None, timeout=None, **kwargs):
        counter["calls"] += 1
        if mode == "http_500":
            response = requests_module.Response()
            response.status_code = 500
            response._content = b"upstream failure"
            return response
        if mode == "malformed_json":
            response = requests_module.Response()
            response.status_code = 200
            response._content = b"not json at all"
            response.headers["Content-Type"] = "application/json"
            return response
        if mode == "read_timeout":
            raise exceptions.ReadTimeout("read timed out")
        if mode == "connect_timeout":
            raise exceptions.ConnectTimeout("connect timed out")
        if mode == "connection_error":
            raise exceptions.ConnectionError("Connection refused by peer")
        if mode == "connection_error_read_timed_out":
            raise exceptions.ConnectionError("HTTPSConnectionPool: Read timed out.")
        raise AssertionError(f"unknown failure mode {mode}")

    return fake_post


def _ordering_post(requests_module, order: list[str]):
    """Delay the first prompt so completion order differs from input order."""
    def fake_post(self, url, headers=None, data=None, timeout=None, **kwargs):
        payload = json.loads(data)
        message = payload["messages"][-1]["content"]
        if message == "first":
            time.sleep(0.25)
        order.append(message)
        return _json_response(
            requests_module, {"choices": [{"message": {"content": message.upper()}}]}
        )

    return fake_post


def _build_serving_api(
    requests_module,
    serving_factory: Callable[..., Any],
    format_impl: Callable[[Any, Any], Any],
    aggregate_impl: Callable[[list[dict[str, Any]]], Any],
    *,
    patch_target: Any,
) -> dict[str, Any]:
    """Assemble the serving half of an api bundle for one implementation."""
    session_cls = patch_target

    @contextlib.contextmanager
    def capture_requests() -> Iterator[list[dict[str, Any]]]:
        recorded: list[dict[str, Any]] = []
        original = session_cls.post
        session_cls.post = _install_capture(requests_module, recorded)
        try:
            yield recorded
        finally:
            session_cls.post = original

    def make_serving(*, key_present: bool = True, **kwargs):
        previous = os.environ.get(KEY_ENV)
        if key_present:
            os.environ[KEY_ENV] = KEY_VALUE
        else:
            os.environ.pop(KEY_ENV, None)
        try:
            return serving_factory(**kwargs)
        finally:
            if previous is None:
                os.environ.pop(KEY_ENV, None)
            else:
                os.environ[KEY_ENV] = previous

    def failure_probe(mode: str, *, max_retries: int) -> dict[str, Any]:
        counter = {"calls": 0}
        original = session_cls.post
        session_cls.post = _failure_post(requests_module, mode, counter)
        try:
            started = time.time()
            serving = make_serving(max_retries=max_retries)
            result = serving.generate_from_input(["prompt"])
            elapsed = time.time() - started
        finally:
            session_cls.post = original
        return {
            "result": result,
            "attempts": counter["calls"],
            # Bucketed, since the reference sleeps 2**i between attempts and the
            # exact wall time is not reproducible.
            "slept_at_least_4s": elapsed >= 4.0,
        }

    def ordering_probe() -> dict[str, Any]:
        order: list[str] = []
        original = session_cls.post
        session_cls.post = _ordering_post(requests_module, order)
        try:
            serving = make_serving(max_workers=4)
            results = serving.generate_from_input(["first", "second", "third"])
        finally:
            session_cls.post = original
        return {"results": results, "completion_order": list(order)}

    return {
        "make_serving": make_serving,
        "format_response": format_impl,
        "aggregate_sse": aggregate_impl,
        "capture_requests": capture_requests,
        "failure_probe": failure_probe,
        "ordering_probe": ordering_probe,
    }


# --------------------------------------------------------------------------
# reference: real open-dataflow 1.0.10 + DataFlow-Evolver compat
# --------------------------------------------------------------------------

def _load_module_from_path(name: str, path: Path):
    """Import a module by path without leaving bytecode next to the source.

    The compat modules live in a read-only checkout, so writing ``__pycache__``
    into it would modify a repository this project must not touch.
    """
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def _load_reference_serving_module():
    """Import the reference serving module without its heavy siblings.

    ``dataflow.serving.__init__`` imports vllm/transformers-backed classes that
    are irrelevant here and need GPU-scale dependencies, so load the one module
    by path under its real package name.
    """
    import dataflow

    full_name = "dataflow.serving.api_llm_serving_request"
    if full_name in sys.modules:
        return sys.modules[full_name]
    package_dir = Path(dataflow.__file__).parent / "serving"
    if "dataflow.serving" not in sys.modules:
        package = types.ModuleType("dataflow.serving")
        package.__path__ = [str(package_dir)]
        package.__package__ = "dataflow.serving"
        sys.modules["dataflow.serving"] = package
    return _load_module_from_path(full_name, package_dir / "api_llm_serving_request.py")


def reference_compat_dir() -> Path:
    """Locate the read-only DataFlow-Evolver checkout holding compat/."""
    override = os.environ.get("DFE_BASELINE_REPO")
    candidates = [Path(override)] if override else []
    candidates.append(Path(__file__).resolve().parents[2].parent / "DataFlow-Evolver")
    for candidate in candidates:
        if (candidate / "dataflow_evolver" / "compat" / "storage.py").is_file():
            return candidate
    raise RuntimeError(
        "DataFlow-Evolver baseline checkout not found; set DFE_BASELINE_REPO to it"
    )


def reference_api() -> dict[str, Any]:
    """Bundle the effective baseline: the real package plus the compat fixes."""
    import requests

    from dataflow.core import LLMServingABC, OperatorABC
    from dataflow.pipeline import (
        BatchedPipelineABC,
        PipelineABC,
        StreamBatchedPipelineABC,
    )
    from dataflow.utils.storage import (
        BatchedFileStorage,
        FileStorage,
        StreamBatchedFileStorage,
    )

    serving_module = _load_reference_serving_module()
    APILLMServing_request = serving_module.APILLMServing_request

    compat_root = reference_compat_dir()
    compat_storage = _load_module_from_path(
        "dfe_compat_storage", compat_root / "dataflow_evolver/compat/storage.py"
    )
    compat_serving = _load_module_from_path(
        "dfe_compat_openai_serving",
        compat_root / "dataflow_evolver/compat/openai_serving.py",
    )
    compat_storage.install_file_storage_null_normalization()
    compat_serving.install_api_llm_reasoning_compat()

    def serving_factory(**kwargs):
        kwargs.setdefault("api_url", API_URL)
        kwargs.setdefault("key_name_of_api_key", KEY_ENV)
        kwargs.setdefault("model_name", MODEL)
        return APILLMServing_request(**kwargs)

    def format_response(payload, enable_thinking):
        serving = APILLMServing_request.__new__(APILLMServing_request)
        serving.configs = (
            {} if enable_thinking == "__absent__" else {"enable_thinking": enable_thinking}
        )
        return serving.format_response(payload)

    def aggregate_sse(chunks):
        request = types.SimpleNamespace(
            body=json.dumps(
                {"messages": [{"role": "user", "content": "q"}], "stream": True}
            ).encode("utf-8"),
            url=API_URL,
        )
        response = _sse_response(requests, chunks)
        return compat_serving.normalize_streaming_chat_response(request, response).json()

    api = {
        "OperatorABC": OperatorABC,
        "LLMServingABC": LLMServingABC,
        "PipelineABC": PipelineABC,
        "BatchedPipelineABC": BatchedPipelineABC,
        "StreamBatchedPipelineABC": StreamBatchedPipelineABC,
        "FileStorage": FileStorage,
        "BatchedFileStorage": BatchedFileStorage,
        "StreamBatchedFileStorage": StreamBatchedFileStorage,
    }
    api.update(
        _build_serving_api(
            requests,
            serving_factory,
            format_response,
            aggregate_sse,
            patch_target=requests.sessions.Session,
        )
    )
    return api


# --------------------------------------------------------------------------
# active implementation: rsi.framework
# --------------------------------------------------------------------------

def framework_api() -> dict[str, Any]:
    """Bundle the active ``rsi.framework`` runtime."""
    import requests

    from rsi.framework import (
        BatchedFileStorage,
        BatchedPipelineABC,
        FileStorage,
        LLMServingABC,
        OperatorABC,
        PipelineABC,
        PipelineLLMServing,
        StreamBatchedFileStorage,
        StreamBatchedPipelineABC,
    )
    from rsi.framework.io.serving import aggregate_chat_stream, format_chat_response

    def serving_factory(**kwargs):
        kwargs.setdefault("api_url", API_URL)
        kwargs.setdefault("key_name_of_api_key", KEY_ENV)
        kwargs.setdefault("model_name", MODEL)
        return PipelineLLMServing(**kwargs)

    def format_response(payload, enable_thinking):
        state = None if enable_thinking == "__absent__" else enable_thinking
        return format_chat_response(payload, enable_thinking=state)

    def aggregate_sse(chunks):
        lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
        lines.append("data: [DONE]")
        body = ("\n\n".join(lines) + "\n\n").encode("utf-8")
        return aggregate_chat_stream(body)

    api = {
        "OperatorABC": OperatorABC,
        "LLMServingABC": LLMServingABC,
        "PipelineABC": PipelineABC,
        "BatchedPipelineABC": BatchedPipelineABC,
        "StreamBatchedPipelineABC": StreamBatchedPipelineABC,
        "FileStorage": FileStorage,
        "BatchedFileStorage": BatchedFileStorage,
        "StreamBatchedFileStorage": StreamBatchedFileStorage,
    }
    api.update(
        _build_serving_api(
            requests,
            serving_factory,
            format_response,
            aggregate_sse,
            patch_target=requests.sessions.Session,
        )
    )
    return api
