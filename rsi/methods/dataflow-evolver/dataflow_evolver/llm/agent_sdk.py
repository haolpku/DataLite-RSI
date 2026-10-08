"""Claude Agent SDK backend for PipelineAgent.

PipelineAgent 通过它启动一次无头 Agent 运行，让 SDK 内的 Claude 借助本地技能与文件
工具把完整 DataFlow 流水线写到节点 workspace。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataflow_evolver.llm.agent_credentials import (
    backend_config,
    backend_value,
    resolve_agent_endpoint,
)
from dataflow_evolver.llm.session_state import (
    finalize_agent_session,
    prepare_agent_session_store,
)
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.agent_sdk")


class AgentInfraError(RuntimeError):
    """Agent runtime 基础设施级失败，区别于“运行完成但产物不合格”。

    `retryable`：True=瞬时错误（重试可能恢复，如 429/超时/连接）；False=永久错误
    （如 CLI 缺失、鉴权失败，重试无益）。主循环据此决定「退避重试本轮」还是「终止运行」。
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


# 判定为「瞬时、可重试」的错误文本标记（对错误信息 + 最近若干条助手文本做小写子串匹配）。
_RETRYABLE_ERROR_TYPES = {"Timeout", "CLIConnectionError"}
_TRANSIENT_MARKERS = (
    "429", "request rejected", "temporary issues", "temporarily unavailable",
    "temporarily", "overloaded", "high demand", "rate limit", "rate_limit", "too many requests",
    "502", "503", "504", "internal server error", "bad gateway",
    "service unavailable", "gateway timeout", "timed out", "timeout",
    "connection reset", "connection error", "connection refused", "econnreset",
    "stream disconnected", "upstream request failed",
)

_SYSTEM_PROMPT_APPEND_TEMPLATE = """
You are the DataFlow-Evolver Pipeline Agent. Activate the project skill
`dataflow-evolver-pipeline` and treat it as the authoritative API/runtime
contract. Generate custom local operators and one bounded pipeline for the task.

Use this specialized skill, the supplied task context, parent artifacts,
diagnostics, and dynamic serving configuration as the complete contract. Do not execute or benchmark
the generated pipeline; the framework owns compile and execution.
"""

def default_system_prompt_append(backend: str) -> str:
    """Return shared PipelineAgent guidance for every supported backend."""
    return _SYSTEM_PROMPT_APPEND_TEMPLATE


_DIAGNOSTIC_SYSTEM_PROMPT_APPEND_TEMPLATE = """
你是 DataFlow-Evolver 的流水线执行诊断 Agent。请以只读方式检查当前 iteration
中的 pipeline.py、operators/、decision.json 和 execution_observation.json。
不要创建、修改、删除或执行任何项目文件，也不要读取完整 JSONL 数据集；只使用框架提供的
有界样本和确定性观测。对照声明的算子职责、筛选规则和配额，检查实际步骤行数、保留率、
字段和低基数字段分布是否一致。你的输出只用于后续 PipelineAgent 的机制性反馈，不参与
Review 分数、passed 状态或 incumbent 选择。严格按调用 prompt 要求返回 JSON。
"""


def default_diagnostic_system_prompt_append(backend: str) -> str:
    """Return independent read-only guidance for the Call B diagnostic runtime."""
    return _DIAGNOSTIC_SYSTEM_PROMPT_APPEND_TEMPLATE



@dataclass
class AgentRunResult:
    """Claude、OpenCode 与 Codex runtime 共用的单次运行结果。"""

    success: bool
    result_text: str = ""                  # ResultMessage.result 的最终文本
    assistant_texts: list[str] = field(default_factory=list)  # 过程中的助手文本块
    error: str | None = None
    error_type: str | None = None          # 异常类名，便于上层分类处理
    exit_code: int | None = None           # ProcessError 时填充
    retryable: bool = False                # 失败是否为瞬时可重试错误（供上层区分基础设施失败类型）
    session_id: str | None = None          # 本次运行的会话 id（用于失败后 resume 续跑，见 run_pipeline_agent）
    # Tool trace for debugging PipelineAgent behavior.
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    # Backend-native per-turn usage normalized to a JSON-safe mapping when available.
    usage: dict[str, Any] | None = None


def _truncate_arg(value: Any, maxlen: int = 600) -> Any:
    if isinstance(value, str):
        return value if len(value) <= maxlen else value[:maxlen] + f"…({len(value)} chars)"
    if isinstance(value, dict):
        return {str(key): _truncate_arg(item, maxlen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_arg(item, maxlen) for item in value[:20]]
    return value


def _record_tool_uses(iteration_dir: str | Path, uses: list[dict[str, Any]], phase: str) -> None:
    if not uses:
        return
    path = Path(iteration_dir) / "agent_tools.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for use in uses:
                handle.write(
                    json.dumps(
                        {"phase": phase, **use},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("写 Agent 工具日志失败（%s）：%s", path, exc)


def _agent_env(agent_cfg: Config, backend: str = "") -> dict[str, str]:
    """给 Agent 子进程叠加环境变量，并与框架执行器使用同一 Python 环境。

    Agent 只写产物、不自行执行流水线；统一 PATH 仍保证其读取依赖元数据或运行普通开发工具时
    看到的环境与框架随后执行 artifact/compile/full-run 检查时一致。
    返回值由具体 runtime 叠加到 ``os.environ``。
    """
    bin_dir = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    env = {"PATH": f"{bin_dir}{os.pathsep}{path}" if path else bin_dir}
    # 配置可追加/覆盖任意 env（如需给 agent 额外变量）；Config.get 对 dict 会回包成 Config
    extra = agent_cfg.get("env", {})
    extra = extra.to_dict() if isinstance(extra, Config) else (extra or {})
    env.update({str(k): str(v) for k, v in extra.items()})
    if backend:
        backend_extra = backend_config(agent_cfg, backend).get("env", {}) or {}
        if isinstance(backend_extra, Config):
            backend_extra = backend_extra.to_dict()
        env.update({str(k): str(v) for k, v in backend_extra.items()})
    return env


def _build_options(
    agent_cfg: Config,
    cwd: str,
    resume: str | None = None,
    session_env: dict[str, str] | None = None,
    system_prompt_append: str | None = None,
    enable_project_skills: bool = True,
) -> Any:
    """据 agent 配置段构造 ClaudeAgentOptions。

    - tools：用 claude_code 预设放开**全部内置工具**（不再用 allowed_tools 白名单限制）。
    - system_prompt：保留 claude_code 预设，仅 **append** 我们的 DataFlow 指令（不覆盖系统提示）。
    - mcp_servers：不挂载 DataFlow operator MCP；Pipeline API 契约由专用 skill 提供。
    - max_turns：单次 SDK 运行的最大轮数（默认 50）。
    - resume：非空则续跑指定会话（载入其历史对话），用于瞬时失败后接着上次进度跑，
      不 fork（继续同一会话 id，把新 turn 追加进去）。
    - effort：Claude 努力/思考档位（low|medium|high|xhigh|max）。配置留空则用模型默认（high）。
    - thinking：显式传 adaptive 思考配置。opus-4-8 等新模型**只支持 adaptive thinking + effort**，
      不接受老式 thinking.enabled（budget）。这里显式指定 {"type":"adaptive"}，从协议层复刻实测
      可用的请求格式，从而不受环境里 CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1（会退化成 thinking.enabled
      → Bedrock 400）影响；即项目自带正确配置，不依赖全局/外部 env。配置 thinking: none 可关闭该显式指定。
    - env：叠加 PATH，使 Agent 开发环境与框架执行器使用的 sys.executable 一致（见 _agent_env）。
    对外使用 provider-neutral Agent 凭据配置；runtime 内部映射为 Claude SDK
    要求的 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL，并保留旧环境变量回退。
    """
    from claude_agent_sdk import ClaudeAgentOptions

    append_text = system_prompt_append or default_system_prompt_append("claude")
    model = backend_value(agent_cfg, "claude", "model", default="claude-opus-4-8")
    effort = backend_value(agent_cfg, "claude", "effort", default="max") or None
    thinking_mode = (agent_cfg.get("thinking", "adaptive") or "adaptive").lower()
    thinking = {"type": "adaptive"} if thinking_mode == "adaptive" else None
    claude_cfg = backend_config(agent_cfg, "claude")
    permission_mode = claude_cfg.get("permission_mode", "bypassPermissions")
    if bool(agent_cfg.get("read_only", False)):
        permission_mode = "plan"
    env = _agent_env(agent_cfg, "claude")
    endpoint = resolve_agent_endpoint(agent_cfg, "claude")
    if endpoint.api_key:
        env["ANTHROPIC_API_KEY"] = endpoint.api_key
    if endpoint.base_url:
        env["ANTHROPIC_BASE_URL"] = endpoint.base_url
    if session_env:
        env.update(session_env)
    return ClaudeAgentOptions(
        cwd=cwd,
        model=model,
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
        permission_mode=permission_mode,
        setting_sources=(
            list(agent_cfg.get("setting_sources", ["project"]))
            if enable_project_skills
            else []
        ),
        skills="all" if enable_project_skills else [],
        max_turns=agent_cfg.get("max_turns", 50),
        mcp_servers={},
        resume=resume,
        effort=effort,
        thinking=thinking,
        env=env,
    )


async def _run(prompt: str, options: Any) -> AgentRunResult:
    from claude_agent_sdk import (
        AssistantMessage,
        CLIConnectionError,
        CLIJSONDecodeError,
        CLINotFoundError,
        ProcessError,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    assistant_texts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    result_text = ""
    turn = 0
    # 会话 id：init/assistant/result 等消息都可能携带；捕获用于失败后 resume 续跑。
    # 用 nonlocal-ish 闭包变量以便异常路径也能带出已捕获的 session_id。
    session_id: str | None = None
    try:
        async for msg in query(prompt=prompt, options=options):
            sid = getattr(msg, "session_id", None)
            if sid:
                session_id = sid
            if isinstance(msg, AssistantMessage):
                turn += 1
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        assistant_texts.append(block.text)
                        preview = block.text[:120].replace("\n", " ")
                        logger.debug("Agent SDK turn %d: %s", turn, preview)
                    elif isinstance(block, ToolUseBlock):
                        logger.debug("Agent SDK turn %d: tool_use %s", turn, block.name)
                        tool_uses.append({
                            "turn": turn,
                            "name": block.name,
                            "input": _truncate_arg(getattr(block, "input", None)),
                        })
            elif isinstance(msg, ResultMessage):
                result_text = msg.result or ""
    except (
        CLINotFoundError, ProcessError, CLIJSONDecodeError, CLIConnectionError, Exception,
    ) as exc:
        # 关键：CLI 在 turn 中途 API 报错时，真实错误（如 "API Error: 400 ... invalid beta flag"）
        # 是作为 assistant 文本返回的，而 SDK 抛出的 ProcessError 常被替换成
        # "Claude Code returned an error result: <subtype>"（subtype 可能是误导性的 "success"）。
        # 故这里把已收到的末条助手文本并入错误、并把 assistant_texts 带出，供 _is_retryable 判定，
        # 也让日志能显示真实原因，而非无信息的 "success"。
        err = str(exc)
        tail = assistant_texts[-1].strip() if assistant_texts else ""
        if tail and tail not in err:
            err = f"{err}｜CLI 末条输出: {tail[:800]}"
        etype = "ProcessError" if isinstance(exc, ProcessError) else type(exc).__name__
        return AgentRunResult(
            False, error=err, error_type=etype,
            exit_code=getattr(exc, "exit_code", None),
            assistant_texts=assistant_texts, session_id=session_id,
            tool_uses=tool_uses,
        )

    return AgentRunResult(
        success=True,
        result_text=result_text,
        assistant_texts=assistant_texts,
        session_id=session_id,
        tool_uses=tool_uses,
    )


# query 在模块级惰性绑定，方便测试时打桩
def query(*args: Any, **kwargs: Any) -> Any:  # noqa: D401
    from claude_agent_sdk import query as _query

    return _query(*args, **kwargs)


def _is_retryable(result: AgentRunResult) -> bool:
    """据错误类型与错误/助手文本判断该失败是否为瞬时、可重试错误。"""
    if result.error_type in _RETRYABLE_ERROR_TYPES:
        return True
    haystack = " ".join([result.error or ""] + result.assistant_texts[-3:]).lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


async def _run_with_timeout(prompt: str, options: Any, timeout: float) -> AgentRunResult:
    """给单次 SDK 运行套超时；<=0 表示不限时。超时会取消协程并向上抛 TimeoutError。"""
    if timeout and timeout > 0:
        return await asyncio.wait_for(_run(prompt, options), timeout)
    return await _run(prompt, options)


# 续跑时发给 agent 的简短提示（原始任务 prompt 已在会话历史里，无需重发整段）。
_RESUME_NUDGE = (
    "The previous run was interrupted by a transient API error. "
    "Continue from where you left off and finish the task: write a working "
    "pipeline.py and decision.json into the target iteration directory. Do not execute "
    "the pipeline; the framework owns artifact validation, compile preflight, and execution."
)


def run_pipeline_agent(
    prompt: str,
    cwd: str,
    agent_cfg: Config,
    tool_log_dir: str | None = None,
    phase: str = "",
    system_prompt_append: str | None = None,
    disable_project_skills: bool = False,
) -> AgentRunResult:
    """Run one PipelineAgent session with timeout, retry, and optional resume.

    `tool_log_dir` 非空时，**每次尝试**结束都把该次的工具调用追加到该目录的 agent_tools.jsonl。
    每次尝试的工具调用都追加落盘，便于复盘失败会话。

    - 瞬时错误（429/5xx/连接/超时）→ 指数退避重试 `max_retries` 次；
    - 续跑（resume_on_retry，默认开）：若上次运行已建立会话（拿到 session_id），重试时
      `resume` 该会话、只发一句简短续跑提示，接着上次进度跑，避免重跑已完成的 turn 白烧 token；
      若上次连会话都没建起（如 turn 1 就 429，无 session_id）则退回全新重跑；
    - 续跑遇到非瞬时错误（如会话丢失/resume 本身出错）→ 丢弃会话、按可重试处理，下次全新重跑，
      绝不因 resume 自身问题把结果判成不可重试（避免上层误熔断）；
    - 永久错误（CLI 缺失/鉴权等，且发生在全新运行）或重试用尽 → 返回 success=False（不抛异常）；
    - 返回结果的 `retryable` 字段标注最终失败是否为瞬时错误，供上层区分处理。
    """
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 3600))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))
    session_store = prepare_agent_session_store(
        agent_cfg,
        backend="claude",
        cwd=cwd,
        tool_log_dir=tool_log_dir,
    )
    session_env = session_store.environment() if session_store is not None else None

    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 SDK 运行", error_type="NoAttempt")
    for attempt in range(1, max_retries + 1):
        used_resume = resume_enabled and resume_id is not None
        options = _build_options(
            agent_cfg,
            cwd,
            resume=resume_id if used_resume else None,
            session_env=session_env,
            system_prompt_append=system_prompt_append,
            enable_project_skills=not disable_project_skills,
        )
        cur_prompt = _RESUME_NUDGE if used_resume else prompt
        logger.info(
            "启动 Agent SDK 运行（尝试 %d/%d，%s）：cwd=%s",
            attempt, max_retries,
            f"resume={resume_id}" if used_resume else "全新会话", cwd,
        )
        try:
            result = asyncio.run(_run_with_timeout(cur_prompt, options, timeout))
        except (asyncio.TimeoutError, TimeoutError):
            result = AgentRunResult(
                False,
                error=f"Agent SDK 运行超过 {timeout:.0f}s 超时",
                error_type="Timeout",
            )

        if tool_log_dir:
            label = phase if attempt == 1 else f"{phase}#retry{attempt - 1}"
            _record_tool_uses(tool_log_dir, result.tool_uses, phase=label)

        # 记录会话 id（成功/失败都可能带出），供下次重试续跑
        if result.session_id:
            resume_id = result.session_id

        if result.success:
            return finalize_agent_session(
                session_store, phase=phase, result=result
            )

        result.retryable = _is_retryable(result)

        # 续跑遇到非瞬时错误：多半是会话丢失 / resume 本身出错。丢弃会话改全新重跑，
        # 并按可重试处理，避免把 resume 的问题误判成不可重试而让上层熔断。
        if used_resume and not result.retryable:
            logger.warning(
                "resume 续跑遇到非瞬时错误（%s），丢弃会话改为全新重跑：%s",
                result.error_type, result.error,
            )
            resume_id = None
            result.retryable = True

        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Agent SDK 第 %d/%d 次失败（%s，可重试%s），%.0fs 后重试：%s",
                attempt, max_retries, result.error_type,
                "，将续跑" if (resume_enabled and resume_id is not None) else "",
                delay, result.error,
            )
            time.sleep(delay)
            continue

        logger.error(
            "Agent SDK 运行失败 [%s]（retryable=%s，已尝试 %d 次）：%s",
            result.error_type, result.retryable, attempt, result.error,
        )
        return finalize_agent_session(session_store, phase=phase, result=result)
    return finalize_agent_session(session_store, phase=phase, result=result)
