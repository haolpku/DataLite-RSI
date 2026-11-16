"""Build and refine DataFlow pipelines from direct ReviewAgent feedback.

它不直接产出文本，而是通过配置的 AgentRuntime 启动无头编码 Agent，让 Agent 只依据专用
dataflow-evolver-pipeline 技能和本轮上下文，把完整可运行的流水线写到 iteration workspace；
然后框架读取 pipeline.py + decision.json，并负责确定性的 artifact、compile 与执行检查。

注意：PipelineAgent 不继承 AgentABC（那是非-SDK 辅助 Agent 的基类）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rsi.framework.evolution.evaluation.downstream_eval import DownstreamEvaluationResult
from rsi.framework.evolution.corpus import InputCorpus
from rsi.framework.evolution.models import (
    PipelineConfig,
    ReviewResult,
    TaskSpec,
)
from rsi.framework.evolution.providers.agent_runtime import AgentRuntime, build_agent_runtime
from rsi.framework.evolution.providers.agent_sdk import AgentInfraError
from rsi.framework.evolution.prompts import (
    build_pipeline_prompt,
    build_selfcorrect_prompt,
)
from rsi.framework.evolution.utils.config import Config
from rsi.framework.evolution.utils.logging import get_logger


class PipelineAgent:
    name = "pipeline"

    def __init__(
        self,
        agent_cfg: Config,
        workspace_dir: str,
        corpus: InputCorpus,
        llm_block: str = "",
        runtime: AgentRuntime | None = None,
    ) -> None:
        self.agent_cfg = agent_cfg
        self.workspace_dir = Path(workspace_dir).resolve()
        self.corpus = corpus
        self.pipeline_filename = agent_cfg.get("pipeline_filename", "pipeline.py")
        self.llm_block = llm_block
        self.backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
        self.runtime = runtime or build_agent_runtime(agent_cfg)
        self.logger = get_logger(f"agents.{self.name}")

    # ---- Initial candidate ----
    def generate_initial(
        self,
        task: TaskSpec,
        iteration_dir: Path,
        downstream_feedback: DownstreamEvaluationResult | None = None,
    ) -> PipelineConfig | None:
        """Build the first pipeline over the fixed raw corpus."""
        iteration_dir = iteration_dir.resolve()
        iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_pipeline_prompt(
            task=task,
            corpus=self.corpus,
            iteration_dir=str(iteration_dir),
            llm_block=self.llm_block,
            downstream_feedback=downstream_feedback,
            is_init=True,
        )
        return self._run_and_assemble(prompt, iteration_dir, phase="generate_initial")

    # ---- Next review-guided candidate ----
    def generate(
        self,
        task: TaskSpec,
        iteration_dir: Path,
        parent_code: str | None = None,
        parent_rationale: str | None = None,
        parent_iteration_dir: Path | None = None,
        parent_review: ReviewResult | None = None,
        execution_observation_path: str | None = None,
        pipeline_diagnostic: Any | None = None,
        downstream_feedback: DownstreamEvaluationResult | None = None,
        evolution_history: list[dict[str, Any]] | None = None,
    ) -> PipelineConfig | None:
        """Challenge the incumbent using its review and compact attempt history."""
        iteration_dir = iteration_dir.resolve()
        iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_pipeline_prompt(
            task=task,
            corpus=self.corpus,
            iteration_dir=str(iteration_dir),
            llm_block=self.llm_block,
            parent_code=parent_code,
            parent_rationale=parent_rationale,
            parent_iteration_dir=(
                str(parent_iteration_dir.resolve())
                if parent_iteration_dir is not None
                else None
            ),
            parent_review=parent_review,
            execution_observation_path=execution_observation_path,
            pipeline_diagnostic=pipeline_diagnostic,
            downstream_feedback=downstream_feedback,
            evolution_history=evolution_history,
        )
        return self._run_and_assemble(prompt, iteration_dir)

    # ---- 自修正（执行崩溃后重试）----
    def self_correct(
        self,
        iteration_dir: Path,
        traceback: str,
        diagnostics: str = "",
        failure_stage: str = "execution",
        log_paths: dict[str, str] | None = None,
    ) -> PipelineConfig | None:
        iteration_dir = iteration_dir.resolve()
        prompt = build_selfcorrect_prompt(
            str(iteration_dir),
            traceback,
            diagnostics,
            failure_stage=failure_stage,
            log_paths=log_paths,
        )
        return self._run_and_assemble(
            prompt,
            iteration_dir,
            warn_stale_decision=True,
            phase=f"repair_{failure_stage}",
        )

    # ---- 内部：跑 SDK + 读产物 ----
    def _run_and_assemble(
        self,
        prompt: str,
        iteration_dir: Path,
        warn_stale_decision: bool = False,
        phase: str = "generate",
    ) -> PipelineConfig | None:
        iteration_dir = iteration_dir.resolve()
        decision_path = iteration_dir / "decision.json"
        fp_before = _fingerprint(decision_path) if warn_stale_decision else None
        result = self.runtime.run(
            prompt,
            cwd=str(self.workspace_dir),
            agent_cfg=self.agent_cfg,
            tool_log_dir=str(iteration_dir),
            phase=phase,
        )
        if not result.success:
            # Infrastructure failure, distinct from an invalid pipeline artifact.
            self.logger.error(
                "PipelineAgent runtime 运行失败（基础设施级，retryable=%s）：%s",
                result.retryable, result.error,
            )
            raise AgentInfraError(result.error or "Agent runtime 运行失败", retryable=result.retryable)

        pipeline_path = iteration_dir / self.pipeline_filename
        if not pipeline_path.exists():
            self.logger.error("未在 %s 找到生成的流水线文件，视为扩展失败。", pipeline_path)
            return None

        code = pipeline_path.read_text(encoding="utf-8")
        # 自修正后 decision.json 未被改动 = 算子构成/字段流转/设计意图仍是修复前那一版，
        # 而 code 已经变了。只告警不阻断（旧版本可能确实没有结构变化），但要能在日志里查到。
        if warn_stale_decision and _fingerprint(decision_path) == fp_before:
            self.logger.warning(
                "自修正后 %s 未更新：本节点记录的算子/字段流转/设计意图可能与新代码不一致"
                "（下一轮继承的设计说明会不准确）。", decision_path,
            )
        decision = _read_decision(decision_path)
        operators, field_flow, rationale = _parse_decision(decision)
        self.logger.info(
            "PipelineAgent 产出流水线：%s（%d 个算子，flow=%s）",
            pipeline_path, len(operators), field_flow,
        )
        return PipelineConfig(
            operators=operators,
            field_flow=field_flow,
            code=code,
            rationale=rationale,
        )


def _read_decision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fingerprint(path: Path) -> str | None:
    """文件内容指纹（不存在返回 None）。

    用内容而非 mtime 判断"有没有被改"：mtime 有文件系统粒度问题（同一 tick 内的改写测不出），
    而且内容原样重写在语义上本就等于"描述没更新"。
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _first_present(d: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """返回第一个存在且非空的键值（decision.json 由 LLM 自由生成，键名会漂移）。"""
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default


# decision.json 的 schema 虽已写进 prompt，但它由 LLM 生成，键名仍会漂移（实测同一次运行里
# 有节点写 "operators"/"description"/"design_rationale"，有节点写 "ops" 字符串数组）。
# 严格按单一键名会静默丢掉算子列表与 rationale，故对常见同义键宽容匹配。
_OPS_KEYS = ("ops", "operators", "operator_list", "pipeline_ops")
_PURPOSE_KEYS = ("purpose", "description", "desc", "role")
_RATIONALE_KEYS = ("reason", "rationale", "design_rationale", "reasoning")
_FLOW_KEYS = ("field_flow", "fields_flow", "flow")


def _parse_decision(decision: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """把 PipelineAgent 的决策 JSON 归一化为 (operators, field_flow, rationale)。

    ops 可能是字符串列表或对象列表，统一成 [{"name", "custom", "purpose", "params"}]。
    """
    raw_ops = _first_present(decision, _OPS_KEYS, default=[])
    operators: list[dict[str, Any]] = []
    if isinstance(raw_ops, dict):        # 少数情况写成 {算子名: {...}} 映射
        raw_ops = [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in raw_ops.items()]
    if isinstance(raw_ops, (list, tuple)):
        for op in raw_ops:
            if isinstance(op, str):
                operators.append({"name": op, "custom": True, "purpose": "", "params": {}})
            elif isinstance(op, dict):
                operators.append({
                    "name": _first_present(op, ("name", "operator", "class"), "?"),
                    "custom": op.get("custom", True),
                    "purpose": _first_present(op, _PURPOSE_KEYS),
                    "params": op.get("params", {}),
                })
    field_flow = _first_present(decision, _FLOW_KEYS)
    rationale = _first_present(decision, _RATIONALE_KEYS)
    return operators, field_flow, rationale
