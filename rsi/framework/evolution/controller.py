"""Incumbent-challenger raw-data pipeline improvement driven by review feedback.

One challenger is produced per iteration. Only a better candidate replaces the
best-so-far incumbent; compact attempt history discourages repeated failed strategies.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsi.framework.evolution.agents.downstream_attribution import DownstreamAttributionAgent
from rsi.framework.evolution.agents.pipeline_agent import PipelineAgent
from rsi.framework.evolution.agents.pipeline_diagnostic import PipelineDiagnosticAgent
from rsi.framework.evolution.agents.review_agent import ReviewAgent
from rsi.framework.evolution.evaluation import decontamination
from rsi.framework.evolution import corpus as input_corpus
from rsi.framework.evolution.evaluation.downstream_eval import (
    CommandDownstreamEvaluator,
    DownstreamAttributionResult,
    DownstreamEvaluationRequest,
    DownstreamEvaluationResult,
    DownstreamEvaluatorABC,
    result_to_dict,
)
from rsi.framework.evolution.evaluation.embedding_quality import (
    EmbeddingQualityEvaluator,
    OpenAIChatEmbeddingBackend,
    ProxyDatasetConfig,
)
from rsi.framework.evolution.models import PipelineConfig, ReviewResult, TaskSpec
from rsi.framework.evolution.feedback import CandidateEvaluator, normalize_candidate_review
from rsi.framework.core.contracts import InputContract
from rsi.framework.evolution.execution.execution_wrapper import (
    ExecutionResult,
    ExecutionWrapperABC,
    SubprocessExecutionWrapper,
)
from rsi.framework.evolution.providers.agent_runtime import build_pipeline_diagnostic_runtime
from rsi.framework.evolution.providers.serving import build_serving
from rsi.framework.evolution.providers.skill_paths import skill_root_for_backend
from rsi.framework.evolution.utils.config import Config, load_config
from rsi.framework.evolution.utils.logging import get_logger, setup_logging

logger = get_logger("main")


@dataclass
class Candidate:
    """One pipeline/data/review observation in the incumbent feedback loop."""

    iteration: int
    iteration_dir: Path
    config: PipelineConfig
    dataset_path: str | None = None
    review: ReviewResult | None = None
    execution_observation_path: str | None = None
    pipeline_diagnostic: Any | None = None
    execution_error: str | None = None
    failure_stage: str | None = None
    parent_iteration: int | None = None
    accepted_as_incumbent: bool = False
    execution_repairs: int = 0
    wall_time_seconds: float = 0.0

    @property
    def score(self) -> float:
        return self.review.review_score if self.review is not None else 0.0


class ReviewOnlyLoop:
    """A best-so-far incumbent challenged by one generated pipeline per iteration."""

    def __init__(
        self,
        *,
        pipeline_agent: PipelineAgent,
        executor: ExecutionWrapperABC,
        reviewer: CandidateEvaluator,
        task: TaskSpec,
        workspace: Path,
        run_name: str,
        max_iterations: int,
        execution_repairs: int = 1,
        downstream_evaluator: DownstreamEvaluatorABC | None = None,
        downstream_attribution_agent: DownstreamAttributionAgent | None = None,
        downstream_interval: int = 2,
        pipeline_diagnostic_agent: Any | None = None,
        pipeline_diagnostic_enabled: bool = False,
        pipeline_diagnostic_max_prompt_chars: int = 80000,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("loop.max_iterations 必须至少为 1（包含初始流水线）")
        self.pipeline_agent = pipeline_agent
        self.executor = executor
        self.reviewer = reviewer
        self.task = task
        self.workspace = workspace.resolve()
        self.run_name = _safe_run_name(run_name)
        self.max_iterations = max_iterations
        self.execution_repairs = max(0, execution_repairs)
        if downstream_evaluator is not None and downstream_interval < 1:
            raise ValueError("downstream_eval.interval must be at least 1")
        self.downstream_evaluator = downstream_evaluator
        self.downstream_attribution_agent = downstream_attribution_agent
        self.downstream_interval = int(downstream_interval)
        self.pipeline_diagnostic_agent = pipeline_diagnostic_agent
        self.pipeline_diagnostic_enabled = bool(pipeline_diagnostic_enabled)
        self.pipeline_diagnostic_max_prompt_chars = max(
            1000, int(pipeline_diagnostic_max_prompt_chars)
        )
        self.candidates: list[Candidate] = []
        self.downstream_evaluations: list[DownstreamEvaluationResult] = []
        self.latest_downstream_evaluation: DownstreamEvaluationResult | None = None
        self.data_preparation_wall_time_seconds = 0.0
        self.downstream_evaluation_wall_time_seconds = 0.0
        self.total_wall_time_seconds = 0.0

    @property
    def run_dir(self) -> Path:
        return self.workspace / "runs" / self.run_name

    def run(self) -> Candidate | None:
        """Run up to ``max_iterations`` candidates while retaining the best incumbent."""
        run_started = time.perf_counter()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"运行目录已存在且非空：{self.run_dir}。请设置新的 project.run_name，"
                "避免旧 iteration 产物污染本次评审。"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if (
            self.downstream_evaluator is not None
            and self.downstream_evaluator.baseline_enabled
        ):
            self._run_downstream_baseline()

        config = self.pipeline_agent.generate_initial(
            task=self.task,
            iteration_dir=self._iteration_dir(0),
            downstream_feedback=self.latest_downstream_evaluation,
        )
        if config is None:
            logger.error("初始 PipelineAgent 未产出有效流水线。")
            return None

        incumbent: Candidate | None = None
        for iteration in range(self.max_iterations):
            iteration_dir = self._iteration_dir(iteration)
            logger.info("==== Review-only 迭代 %d/%d ====", iteration + 1, self.max_iterations)
            parent_incumbent = incumbent
            candidate = self._execute_and_review(
                iteration,
                iteration_dir,
                config,
                parent_review=(
                    parent_incumbent.review if parent_incumbent is not None else None
                ),
                parent_iteration_dir=(
                    parent_incumbent.iteration_dir if parent_incumbent is not None else None
                ),
            )
            candidate.parent_iteration = (
                parent_incumbent.iteration if parent_incumbent is not None else None
            )
            if _is_better(candidate, incumbent):
                candidate.accepted_as_incumbent = True
                incumbent = candidate
                logger.info(
                    "接受 challenger 为 incumbent：%s score=%.4f",
                    candidate.iteration_dir.name,
                    candidate.score,
                )
            else:
                logger.info(
                    "拒绝 challenger，保留 incumbent=%s",
                    incumbent.iteration_dir.name if incumbent is not None else "none",
                )
            self.candidates.append(candidate)
            self._append_trace(candidate)

            if (
                self.downstream_evaluator is not None
                and (iteration + 1) % self.downstream_interval == 0
                and incumbent is not None
                and incumbent.dataset_path
            ):
                self._run_downstream_checkpoint(iteration + 1, incumbent)

            if iteration + 1 >= self.max_iterations:
                break
            if incumbent is None or incumbent.review is None:
                logger.error("尚无可用 incumbent，无法继续生成 challenger。")
                break

            # Always branch from best-so-far. Rejected attempts enter only a compact
            # decision history, never as inherited pipeline code.
            config = self.pipeline_agent.generate(
                task=self.task,
                iteration_dir=self._iteration_dir(iteration + 1),
                parent_code=incumbent.config.code,
                parent_rationale=incumbent.config.rationale,
                parent_iteration_dir=incumbent.iteration_dir,
                parent_review=incumbent.review,
                execution_observation_path=incumbent.execution_observation_path,
                pipeline_diagnostic=_compact_diagnostic(incumbent.pipeline_diagnostic),
                downstream_feedback=self.latest_downstream_evaluation,
                evolution_history=self._evolution_history(),
            )
            if config is None:
                logger.error("第 %d 轮 PipelineAgent 未产出有效流水线。", iteration + 2)
                break

        best = incumbent or select_best(self.candidates)
        self.total_wall_time_seconds = round(time.perf_counter() - run_started, 6)
        self.downstream_evaluation_wall_time_seconds = round(
            sum(result.wall_time_seconds for result in self.downstream_evaluations), 6
        )
        self.data_preparation_wall_time_seconds = round(
            max(
                0.0,
                self.total_wall_time_seconds
                - self.downstream_evaluation_wall_time_seconds,
            ),
            6,
        )
        if best is not None:
            self._write_manifest(best)
            logger.info(
                "Review-only 最优候选：%s review_score=%.4f passed=%s",
                best.iteration_dir.name,
                best.score,
                best.review.passed if best.review else False,
            )
        return best

    def _iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iteration_{iteration + 1:03d}"

    def _execute_and_review(
        self,
        iteration: int,
        iteration_dir: Path,
        config: PipelineConfig,
        parent_review: ReviewResult | None = None,
        parent_iteration_dir: Path | None = None,
    ) -> Candidate:
        iteration_started = time.perf_counter()
        result: ExecutionResult = self.executor.run(
            config,
            iteration_dir,
            parent_iteration_dir=parent_iteration_dir,
        )
        retries = 0
        while not result.success and retries < self.execution_repairs:
            retries += 1
            logger.info("流水线执行失败，进行第 %d/%d 次代码修复。", retries, self.execution_repairs)
            fixed = self.pipeline_agent.self_correct(
                iteration_dir,
                result.traceback or "",
                result.diagnostics,
                failure_stage=result.failure_stage or "execution",
                log_paths={
                    key: value for key, value in result.artifacts.items()
                    if key.endswith("_log")
                },
            )
            if fixed is None:
                break
            config = fixed
            result = self.executor.run(
                config,
                iteration_dir,
                parent_iteration_dir=parent_iteration_dir,
            )

        if not result.success or not result.dataset_path:
            logger.error("流水线执行最终失败：%s", result.traceback)
            return Candidate(
                iteration=iteration,
                iteration_dir=iteration_dir,
                config=config,
                execution_error=result.traceback or "unknown execution failure",
                failure_stage=result.failure_stage,
                execution_repairs=retries,
                wall_time_seconds=round(time.perf_counter() - iteration_started, 6),
            )

        evidence_path = iteration_dir / "review.jsonl"
        raw_review = self.reviewer.review(
            result.dataset_path,
            self.task,
            evidence_path=str(evidence_path),
            phase=f"iteration_{iteration}",
            parent_review=parent_review,
            usage_dir=str(iteration_dir),
        )
        review = normalize_candidate_review(raw_review)
        review.evaluator_kind = self.task.evaluator_kind
        diagnostic = None
        if self.pipeline_diagnostic_enabled and self.pipeline_diagnostic_agent is not None:
            try:
                diagnostic = self.pipeline_diagnostic_agent.diagnose(
                    iteration_dir=iteration_dir,
                    task=self.task,
                    observation_path=result.observation_path,
                    config=config,
                    review=review,
                    output_path=iteration_dir / "pipeline_diagnostic.json",
                    max_prompt_chars=self.pipeline_diagnostic_max_prompt_chars,
                )
            except Exception as exc:  # noqa: BLE001 - Call B must not stop selection
                logger.exception("Pipeline execution diagnosis failed; continuing: %s", exc)
                diagnostic = {
                    "status": "failed",
                    "error": f"pipeline diagnostic exception: {exc}",
                }
        return Candidate(
            iteration=iteration,
            iteration_dir=iteration_dir,
            config=config,
            dataset_path=result.dataset_path,
            review=review,
            execution_observation_path=result.observation_path,
            pipeline_diagnostic=diagnostic,
            execution_repairs=retries,
            wall_time_seconds=round(time.perf_counter() - iteration_started, 6),
        )

    def _append_trace(self, candidate: Candidate) -> None:
        record = _candidate_record(candidate)
        path = self.run_dir / "iterations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _run_downstream_checkpoint(
        self,
        checkpoint_iteration: int,
        incumbent: Candidate,
    ) -> None:
        """Evaluate the current incumbent after a configured number of static rounds."""
        assert self.downstream_evaluator is not None
        assert incumbent.dataset_path is not None
        checkpoint_dir = (
            self.run_dir / "downstream_eval" / f"checkpoint_{checkpoint_iteration:03d}"
        )
        request = DownstreamEvaluationRequest(
            checkpoint_iteration=checkpoint_iteration,
            incumbent_iteration=incumbent.iteration + 1,
            dataset_path=str(Path(incumbent.dataset_path).resolve()),
            pipeline_path=str((incumbent.iteration_dir / "pipeline.py").resolve()),
            task=self.task,
            expected_benchmarks=list(self.downstream_evaluator.expected_benchmarks),
            bad_cases_per_benchmark=self.downstream_evaluator.bad_cases_per_benchmark,
            stage="periodic",
            training_performed=True,
        )
        logger.info(
            "触发下游 SFT/评测检查点：完成轮次=%d incumbent=%s dataset=%s",
            checkpoint_iteration,
            incumbent.iteration_dir.name,
            incumbent.dataset_path,
        )
        result = self.downstream_evaluator.run(request, checkpoint_dir)
        self._attach_downstream_attribution(result, checkpoint_dir, incumbent)
        self._record_downstream_result(result)

    def _run_downstream_baseline(self) -> None:
        """Evaluate the untrained downstream model before generating iteration zero."""
        assert self.downstream_evaluator is not None
        request = DownstreamEvaluationRequest(
            checkpoint_iteration=0,
            incumbent_iteration=None,
            dataset_path=None,
            pipeline_path=None,
            task=self.task,
            expected_benchmarks=list(self.downstream_evaluator.expected_benchmarks),
            bad_cases_per_benchmark=self.downstream_evaluator.bad_cases_per_benchmark,
            stage="baseline",
            training_performed=False,
        )
        checkpoint_dir = self.run_dir / "downstream_eval" / "checkpoint_000"
        logger.info(
            "触发第 0 轮下游诊断：不训练模型，直接评测外部 runner 的固定诊断集",
        )
        result = self.downstream_evaluator.run(request, checkpoint_dir)
        self._attach_downstream_attribution(result, checkpoint_dir, None)
        self._record_downstream_result(result)

    def _attach_downstream_attribution(
        self,
        result: DownstreamEvaluationResult,
        checkpoint_dir: Path,
        incumbent: Candidate | None,
    ) -> None:
        if (
            not result.succeeded
            or self.downstream_attribution_agent is None
        ):
            return
        try:
            result.attribution = self.downstream_attribution_agent.attribute(
                task=self.task,
                feedback=result,
                pipeline_operators=(
                    incumbent.config.operators if incumbent is not None else []
                ),
                parent_review=(
                    incumbent.review if incumbent is not None else None
                ),
                evidence_path=checkpoint_dir / "attribution.json",
            )
        except Exception as exc:  # noqa: BLE001 — attribution must not stop evolution
            logger.exception("下游 bad case 归因失败，继续保留静态 incumbent：%s", exc)
            result.attribution = DownstreamAttributionResult(
                status="failed",
                error=f"attribution exception: {exc}",
            )

    def _record_downstream_result(self, result: DownstreamEvaluationResult) -> None:
        """Persist one result and compute deltas without mixing it into review_score."""
        if result.succeeded:
            previous = next(
                (item for item in reversed(self.downstream_evaluations) if item.succeeded),
                None,
            )
            if previous is not None:
                result.score_deltas = {
                    name: benchmark.score - previous.benchmarks[name].score
                    for name, benchmark in result.benchmarks.items()
                    if name in previous.benchmarks
                }
            baseline = next(
                (
                    item
                    for item in self.downstream_evaluations
                    if item.succeeded and item.stage == "baseline"
                ),
                None,
            )
            if baseline is not None and result.stage != "baseline":
                result.score_deltas_vs_baseline = {
                    name: benchmark.score - baseline.benchmarks[name].score
                    for name, benchmark in result.benchmarks.items()
                    if name in baseline.benchmarks
                }
        self.downstream_evaluations.append(result)
        self.latest_downstream_evaluation = result
        trace_path = self.run_dir / "downstream_evaluations.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(result_to_dict(result), ensure_ascii=False, default=str) + "\n"
            )
        if result.succeeded:
            scores = ", ".join(
                f"{name}={item.score:.6f}"
                for name, item in result.benchmarks.items()
            )
            logger.info("下游 SFT/评测完成：%s", scores)
        else:
            logger.error("下游 SFT/评测失败；保留静态 incumbent 并继续：%s", result.error)

    def _evolution_history(self) -> list[dict[str, Any]]:
        """Return compact decisions, not prior pipeline code or full reviews."""
        history: list[dict[str, Any]] = []
        for candidate in self.candidates:
            history.append(
                {
                    "iteration": candidate.iteration + 1,
                    "outcome": (
                        "accepted" if candidate.accepted_as_incumbent else "rejected"
                    ),
                    "review_score": candidate.score if candidate.review is not None else None,
                    "passed": candidate.review.passed if candidate.review is not None else False,
                    "operators": [
                        str(operator.get("name", "?"))
                        for operator in candidate.config.operators[:8]
                    ],
                    "decision": _compact_text(candidate.config.rationale, 240),
                    "issues": (
                        [_compact_text(issue, 160) for issue in candidate.review.issues[:2]]
                        if candidate.review is not None
                        else [_compact_text(candidate.execution_error or "execution failed", 160)]
                    ),
                    "execution_observation": candidate.execution_observation_path,
                    "pipeline_diagnostic": _compact_diagnostic(candidate.pipeline_diagnostic),
                }
            )
        return history

    def _write_manifest(self, best: Candidate) -> None:
        manifest = {
            "mode": "pipeline_review",
            "run_name": self.run_name,
            "max_iterations": self.max_iterations,
            "completed_iterations": len(self.candidates),
            "total_execution_repairs": sum(c.execution_repairs for c in self.candidates),
            "data_preparation_wall_time_seconds": self.data_preparation_wall_time_seconds,
            "downstream_evaluation_wall_time_seconds": (
                self.downstream_evaluation_wall_time_seconds
            ),
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "selection_rule": "best-so-far incumbent: passed first, then highest review_score",
            "downstream_evaluation": {
                "enabled": self.downstream_evaluator is not None,
                "interval": self.downstream_interval,
                "completed_checkpoints": len(self.downstream_evaluations),
                "completed_periodic_checkpoints": sum(
                    item.stage == "periodic" for item in self.downstream_evaluations
                ),
                "baseline_enabled": bool(
                    self.downstream_evaluator is not None
                    and self.downstream_evaluator.baseline_enabled
                ),
                "baseline_succeeded": any(
                    item.stage == "baseline" and item.succeeded
                    for item in self.downstream_evaluations
                ),
                "evaluations": [
                    result_to_dict(result) for result in self.downstream_evaluations
                ],
            },
            "best": _candidate_record(best),
            "evolution_history": self._evolution_history(),
            "all_candidates": [_candidate_record(c) for c in self.candidates],
            "manual_next_step": (
                "Use best.dataset_path for final downstream training/evaluation. "
                "Periodic checkpoints are optimization feedback, not the final report."
            ),
        }
        path = self.run_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )


def select_best(candidates: list[Candidate]) -> Candidate | None:
    """Prefer a passed review, then maximize its Review score."""
    reviewed = [candidate for candidate in candidates if candidate.review is not None]
    if not reviewed:
        return None
    return max(reviewed, key=lambda c: (bool(c.review and c.review.passed), c.score))


def _is_better(challenger: Candidate, incumbent: Candidate | None) -> bool:
    if challenger.review is None:
        return False
    if incumbent is None or incumbent.review is None:
        return True
    challenger_rank = (challenger.review.passed, challenger.score)
    incumbent_rank = (incumbent.review.passed, incumbent.score)
    return challenger_rank > incumbent_rank


def build_loop(
    cfg: Config,
    *,
    input_contract: InputContract | None = None,
    candidate_evaluator: CandidateEvaluator | None = None,
    pipeline_agent_override: Any | None = None,
) -> ReviewOnlyLoop:
    """Build the DataFlow loop with optional modality-specific adapters."""
    project = cfg.project
    workspace = Path(project.workspace_dir).resolve()
    backend = str(cfg.agent.get("backend", "claude") or "claude").strip().lower()
    skill_root = skill_root_for_backend(backend)
    if not (workspace / skill_root / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"{backend} backend requires {skill_root}/SKILL.md: {workspace}"
        )

    corpus = input_corpus.discover(project.get("input_path", "") or "")
    if input_contract is not None:
        input_contract.validate_entry(corpus.entry_path)
        if set(input_contract.modalities) & {"image", "video"} and candidate_evaluator is None:
            raise ValueError(
                "image/video DataFlow tasks require a candidate_evaluator; "
                "the default ReviewAgent only assesses structured text"
            )

    task_cfg = cfg.task
    target_schema = _plain_dict(task_cfg.get("target_schema"))
    if not target_schema:
        raise ValueError("task.target_schema 不能为空；入口不会自动猜测输出字段")
    task = TaskSpec(
        task_description="",  # CLI --task is injected in main() after construction.
        target_schema={str(k): str(v) for k, v in target_schema.items()},
        quality_criteria=list(task_cfg.get("quality_criteria", []) or []),
        input_contract=input_contract,
        evaluator_kind="task" if candidate_evaluator is not None else "review_agent",
    )

    serving = build_serving(cfg.llm) if candidate_evaluator is None else None
    if candidate_evaluator is None:
        review_cfg = cfg.review
        sampling_cfg = review_cfg.get("sampling") or Config({})
        criteria_cfg = review_cfg.get("criteria") or Config({})
        four_cfg = review_cfg.get("four_dimensions") or Config({})
        validation_cfg = review_cfg.get("validation") or Config({})
        decon_cfg = review_cfg.get("decontamination") or Config({})
        embedding_quality_cfg = review_cfg.get("embedding_quality") or Config({})
        metrics_cfg = embedding_quality_cfg.get("metrics") or Config({})
        embedding_quality_evaluator, embedding_quality_fields = _build_embedding_quality_evaluator(
            embedding_quality_cfg,
            workspace=workspace,
        )
        reviewer: CandidateEvaluator = ReviewAgent(
            serving,
            contamination=decontamination.build(
                reference_files=list(decon_cfg.get("reference_files", []) or []),
                n=int(decon_cfg.get("ngram_size", decontamination.DEFAULT_NGRAM)),
                max_rate=float(decon_cfg.get("max_rate", decontamination.DEFAULT_MAX_RATE)),
                fields=list(decon_cfg.get("reference_fields", []) or []) or None,
            ),
            four_dimension_thresholds=_plain_dict(
                four_cfg.get("pass_thresholds")
            ) or None,
            sampling_size=int(sampling_cfg.get("size", 80)),
            sampling_seed=sampling_cfg.get("seed", None),
            four_dimension_weights=_plain_dict(four_cfg.get("weights")) or None,
            validation_enabled=_as_bool(validation_cfg.get("enabled", True)),
            max_null_rate=float(validation_cfg.get("max_null_rate", 0.5)),
            embedding_quality_evaluator=embedding_quality_evaluator,
            embedding_quality_fields=embedding_quality_fields,
            embedding_review_score_weight=(
                float(embedding_quality_cfg.get("review_score_weight"))
                if embedding_quality_cfg.get("review_score_weight") is not None
                else None
            ),
            embedding_metric_weights=_plain_dict(
                {
                    name: (metrics_cfg.get(name) or Config({})).get("weight", 0.0)
                    for name in ("das", "vendi", "nearest_neighbor")
                    if _as_bool((metrics_cfg.get(name) or Config({})).get("enabled", False))
                }
            ) or None,
            contamination_fields=(
                list(decon_cfg.get("candidate_fields", []) or []) or None
            ),
            mode=str(review_cfg.get("mode", "four_dimensions") or "four_dimensions"),
            criteria_pass_rate=float(criteria_cfg.get("pass_rate", 0.6)),
        )
    else:
        reviewer = candidate_evaluator

    agent_cfg = cfg.agent

    pipeline_llm = _resolve_pipeline_llm(cfg.llm)
    llm_block, extra_env = _pipeline_runtime_contract(pipeline_llm)
    pipeline_agent = pipeline_agent_override if pipeline_agent_override is not None else PipelineAgent(
        agent_cfg=agent_cfg,
        workspace_dir=str(workspace),
        corpus=corpus,
        llm_block=llm_block,
    )

    diagnostic_cfg = cfg.get("pipeline_diagnostic") or Config({})
    diagnostic_enabled = _as_bool(diagnostic_cfg.get("enabled", False))
    diagnostic_agent = None
    if diagnostic_enabled:
        if not _as_bool(diagnostic_cfg.get("read_only", False)):
            raise ValueError("pipeline_diagnostic.read_only must be true")
        diagnostic_backend = str(diagnostic_cfg.get("backend", "codex") or "codex").lower()
        diagnostic_backend_cfg = diagnostic_cfg.get(diagnostic_backend) or Config({})
        if diagnostic_backend == "codex" and _as_bool(
            diagnostic_backend_cfg.get("bypass_sandbox", False)
        ):
            raise ValueError(
                "pipeline_diagnostic.codex.bypass_sandbox must be false for read-only Call B"
            )
        diagnostic_agent = PipelineDiagnosticAgent(
            agent_cfg=diagnostic_cfg,
            runtime=build_pipeline_diagnostic_runtime(diagnostic_cfg),
        )

    loop_cfg = cfg.loop
    executor = SubprocessExecutionWrapper(
        workspace_dir=str(workspace),
        pipeline_filename=str(agent_cfg.get("pipeline_filename", "pipeline.py")),
        compile_timeout_sec=int(loop_cfg.get("compile_timeout_sec", 120)),
        timeout_sec=int(loop_cfg.get("execution_timeout_sec", 1200)),
        python_exe=str(loop_cfg.get("python_exe", "") or "") or None,
        extra_env=extra_env or None,
        raw_entry_path=corpus.entry_path,
        observation_sample_rows=int(loop_cfg.get("observation_sample_rows", 2)),
        observation_max_sample_chars=int(
            loop_cfg.get("observation_max_sample_chars", 4000)
        ),
    )
    downstream_cfg = cfg.get("downstream_eval") or Config({})
    downstream_evaluator, downstream_interval = _build_downstream_evaluator(
        downstream_cfg
    )
    downstream_attribution_agent = None
    attribution_cfg = downstream_cfg.get("attribution") or Config({})
    if downstream_evaluator is not None and _as_bool(attribution_cfg.get("enabled", True)):
        if serving is None:
            serving = build_serving(cfg.llm)
        downstream_attribution_agent = DownstreamAttributionAgent(
            serving,
            max_retries=int(attribution_cfg.get("max_retries", 2)),
            max_findings=int(attribution_cfg.get("max_findings", 6)),
            max_actions=int(attribution_cfg.get("max_actions", 8)),
        )
    return ReviewOnlyLoop(
        pipeline_agent=pipeline_agent,
        executor=executor,
        reviewer=reviewer,
        task=task,
        workspace=workspace,
        run_name=str(project.get("run_name", "run") or "run"),
        max_iterations=int(loop_cfg.get("max_iterations", 5)),
        execution_repairs=int(loop_cfg.get("execution_repairs", 1)),
        downstream_evaluator=downstream_evaluator,
        downstream_attribution_agent=downstream_attribution_agent,
        downstream_interval=downstream_interval,
        pipeline_diagnostic_agent=diagnostic_agent,
        pipeline_diagnostic_enabled=diagnostic_enabled,
        pipeline_diagnostic_max_prompt_chars=int(
            diagnostic_cfg.get("max_prompt_chars", 80000)
        ),
    )


def _candidate_record(candidate: Candidate) -> dict[str, Any]:
    pipeline_path = candidate.iteration_dir / "pipeline.py"
    return {
        "iteration": candidate.iteration,
        "iteration_dir": str(candidate.iteration_dir),
        "pipeline_path": str(pipeline_path),
        "dataset_path": candidate.dataset_path,
        "execution_error": candidate.execution_error,
        "failure_stage": candidate.failure_stage,
        "execution_observation_path": candidate.execution_observation_path,
        "pipeline_diagnostic": candidate.pipeline_diagnostic,
        "parent_iteration": candidate.parent_iteration,
        "accepted_as_incumbent": candidate.accepted_as_incumbent,
        "execution_repairs": candidate.execution_repairs,
        "wall_time_seconds": candidate.wall_time_seconds,
        "field_flow": candidate.config.field_flow,
        "operators": candidate.config.operators,
        "review": asdict(candidate.review) if candidate.review is not None else None,
    }


def _compact_diagnostic(value: Any) -> dict[str, Any] | None:
    """Keep only small, mechanism-oriented evidence in the next prompt."""
    if not isinstance(value, dict):
        return None
    output: dict[str, Any] = {}
    if value.get("status"):
        output["status"] = value["status"]
    if value.get("summary"):
        output["summary"] = _compact_text(str(value["summary"]), 500)
    findings = value.get("findings")
    if isinstance(findings, list):
        output["findings"] = [
            {
                key: _compact_text(str(item.get(key, "")), 280)
                for key in ("step", "severity", "evidence", "diagnosis", "recommended_action")
                if item.get(key) not in (None, "")
            }
            for item in findings[:3]
            if isinstance(item, dict)
        ]
    mismatches = value.get("contract_mismatches")
    if isinstance(mismatches, list):
        output["contract_mismatches"] = [
            _compact_text(str(item), 300) for item in mismatches[:3]
        ]
    if value.get("error"):
        output["error"] = _compact_text(str(value["error"]), 400)
    return output or None


def _safe_run_name(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError("project.run_name 必须是不含路径分隔符的非空名称")
    return path.name


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return dict(value.to_dict())
    return dict(value or {}) if isinstance(value, dict) else {}


def _build_downstream_evaluator(
    value: Config,
) -> tuple[DownstreamEvaluatorABC | None, int]:
    """Build the optional process-isolated SFT/evaluation adapter."""
    interval = int(value.get("interval", 2))
    if interval < 1:
        raise ValueError("downstream_eval.interval must be at least 1")
    if not _as_bool(value.get("enabled", False)):
        return None, interval

    raw_command = value.get("command", []) or []
    if not isinstance(raw_command, list):
        raise ValueError("downstream_eval.command must be a YAML argv list")
    raw_benchmarks = value.get("benchmarks", []) or []
    if not isinstance(raw_benchmarks, list):
        raise ValueError("downstream_eval.benchmarks must be a YAML list")
    baseline_cfg = value.get("baseline") or Config({})
    return (
        CommandDownstreamEvaluator(
            [str(item) for item in raw_command],
            expected_benchmarks=[str(item) for item in raw_benchmarks],
            bad_cases_per_benchmark=int(value.get("bad_cases_per_benchmark", 3)),
            timeout_sec=int(value.get("timeout_sec", 86400)),
            baseline_enabled=_as_bool(baseline_cfg.get("enabled", False)),
        ),
        interval,
    )


def _build_embedding_quality_evaluator(
    value: Config,
    *,
    workspace: Path,
) -> tuple[EmbeddingQualityEvaluator | None, tuple[str, str]]:
    """Build embedding metrics from their explicit per-metric configuration."""
    metrics_cfg = value.get("metrics") or Config({})
    metric_names = ("das", "vendi", "nearest_neighbor")
    enabled_metrics = [
        name
        for name in metric_names
        if _as_bool((metrics_cfg.get(name) or Config({})).get("enabled", False))
    ]
    metric_weights = {
        name: float((metrics_cfg.get(name) or Config({})).get("weight", 0.0))
        for name in enabled_metrics
    }
    if enabled_metrics and all(weight <= 0.0 for weight in metric_weights.values()):
        raise ValueError("review.embedding_quality.metrics 至少需要一个启用且有正权重的指标")
    candidate_cfg = value.get("candidate") or Config({})
    fields = (
        str(candidate_cfg.get("user_field", "instruction")),
        str(candidate_cfg.get("assistant_field", "output")),
    )
    if not enabled_metrics:
        return None, fields

    sampling_cfg = value.get("sampling") or Config({})
    embedding_cfg = value.get("embedding") or Config({})
    service_cfg = value.get("service") or Config({})
    sample_size = int(sampling_cfg.get("size", 5000))
    sample_seed = int(sampling_cfg.get("seed", 42))
    require_requested_size = _as_bool(sampling_cfg.get("require_requested_size", True))
    cache_value = str(embedding_cfg.get("cache_dir", "") or "").strip()
    cache_dir = Path(cache_value) if cache_value else workspace / "cache" / "embedding"

    das_cfg = metrics_cfg.get("das") or Config({})
    das_enabled = "das" in enabled_metrics
    proxy_cfg = das_cfg.get("proxy") or Config({})
    proxy_path = str(proxy_cfg.get("path", "") or "").strip()
    if das_enabled and not proxy_path:
        raise ValueError("review.embedding_quality.metrics.das.proxy.path 不能为空")

    base_url = str(service_cfg.get("base_url", "") or "").strip()
    model_name = str(service_cfg.get("model_name", "") or "").strip()
    if not base_url:
        raise ValueError("review.embedding_quality.service.base_url 不能为空")
    if not model_name:
        raise ValueError("review.embedding_quality.service.model_name 不能为空")

    proxy = None
    if das_enabled:
        proxy = ProxyDatasetConfig(
            source=str(proxy_cfg.get("source", "local")),
            path=proxy_path,
            name=str(proxy_cfg.get("name", "") or ""),
            split=str(proxy_cfg.get("split", "train")),
            user_field=str(proxy_cfg.get("user_field", "question")),
            assistant_field=str(proxy_cfg.get("assistant_field", "response")),
        )
    embedder = OpenAIChatEmbeddingBackend(
        model_name=model_name,
        base_url=base_url,
        api_key=str(service_cfg.get("api_key", "EMPTY") or "EMPTY"),
        max_concurrent_requests=int(service_cfg.get("max_concurrent_requests", 128)),
        max_retries=int(service_cfg.get("max_retries", 3)),
        truncate_prompt_tokens=int(service_cfg.get("truncate_prompt_tokens", 40960)),
        truncation_side=str(service_cfg.get("truncation_side", "right")),
    )
    mmd_cfg = das_cfg.get("mmd") or Config({})
    estimator = str(mmd_cfg.get("estimator", "biased") or "biased").strip().lower()
    if das_enabled and estimator not in {"biased", "unbiased"}:
        raise ValueError(
            "review.embedding_quality.metrics.das.mmd.estimator 必须是 biased 或 unbiased"
        )
    evaluator = EmbeddingQualityEvaluator(
        enabled_metrics=enabled_metrics,
        proxy=proxy,
        embedder=embedder,
        sample_size=sample_size,
        seed=sample_seed,
        # MMD parameters belong exclusively to DAS. Ignore malformed or
        # absent values when only candidate-only metrics are enabled.
        sigma=float(mmd_cfg.get("sigma", 1.0)) if das_enabled else 1.0,
        biased=(estimator == "biased") if das_enabled else True,
        normalize_embeddings=_as_bool(embedding_cfg.get("normalize_embeddings", True)),
        max_embedding_failure_rate=float(embedding_cfg.get("max_failure_rate", 0.02)),
        cache_dir=cache_dir,
        require_requested_size=require_requested_size,
    )
    return evaluator, fields

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _resolve_pipeline_llm(cfg_llm: Config) -> dict[str, dict[str, Any]]:
    api = cfg_llm.api
    sections = cfg_llm.get("pipeline_llm")

    def resolve(section: Config | None) -> dict[str, Any]:
        def pick(key: str, default: Any = None) -> Any:
            value = section.get(key, None) if section is not None else None
            return api.get(key, default) if value is None else value

        serving_kwargs = section.get("serving_kwargs", {}) if section is not None else {}
        if isinstance(serving_kwargs, Config):
            serving_kwargs = serving_kwargs.to_dict()
        if not isinstance(serving_kwargs, dict):
            raise ValueError("llm.pipeline_llm.*.serving_kwargs must be an object")
        return {
            "api_url": _normalize_pipeline_api_url(pick("api_url")),
            "model_name": pick("model_name"),
            "api_key": pick("api_key", ""),
            "max_workers": pick("max_workers", 10),
            "serving_kwargs": dict(serving_kwargs),
        }

    if sections is None:
        return {"default": resolve(None)}
    raw = sections.to_dict() if isinstance(sections, Config) else sections
    return {
        str(name): resolve(Config(value) if isinstance(value, dict) else None)
        for name, value in raw.items()
    }


def _normalize_pipeline_api_url(value: Any) -> Any:
    """Accept an OpenAI base URL while emitting the endpoint DataFlow posts to."""
    if not isinstance(value, str) or not value.strip():
        return value
    url = value.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def _pipeline_runtime_contract(
    pipeline_llm: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    lines = ["   可用的 Pipeline LLM Serving profiles（profile 名只标识配置，不限定算子职责）："]
    lines.append(
        "   enable_thinking 是三态请求/响应策略：true/false 作为布尔值原样发送；"
        "omit（或 profile 中缺省）表示构造 serving 时省略该 keyword，且按 content-only 解析。"
        "不得把 omit 改写成 false，也不得把 omit 字符串发送给上游 API。"
    )
    env: dict[str, str] = {}
    for profile, item in pipeline_llm.items():
        env_name = f"DF_PIPELINE_API_KEY_{profile.upper()}"
        lines.append(
            f"   [{profile}] model={item['model_name']}, api_url={item['api_url']}, "
            f"key_name_of_api_key={env_name}, max_workers={item['max_workers']}, "
            f"default_serving_kwargs={json.dumps(item.get('serving_kwargs', {}), ensure_ascii=False)}"
        )
        lines.append(
            "      按算子所需的响应契约选择 profile；选定后必须原样使用其 default_serving_kwargs。"
            "不得因算子需求擅自改写 profile 参数。"
        )
        if item.get("api_key"):
            env[env_name] = str(item["api_key"])
    lines.append("   key_name_of_api_key 是环境变量名；禁止把密钥写入 pipeline.py。")
    return "\n".join(lines), env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed raw data + PipelineAgent + ReviewAgent loop")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(args.log_level, Path(cfg.project.workspace_dir) / "run.log")
    loop = build_loop(cfg)
    loop.task.task_description = args.task
    best = loop.run()
    if best is None:
        print("\n没有得到可评审的数据集。")
        return
    print(f"\n最优流水线：{best.iteration_dir / 'pipeline.py'}")
    print(f"最优数据：{best.dataset_path}")
    print(f"Review score：{best.score:.4f}（passed={best.review.passed}）")
    print(f"运行清单：{loop.run_dir / 'manifest.json'}")
    if loop.downstream_evaluations:
        print(
            f"周期性下游 SFT/评测检查点：{len(loop.downstream_evaluations)}；"
            "最终训练和评测仍应基于 manifest 中的 best.dataset_path 单独执行。"
        )
    else:
        print("训练和下游评测未执行；请根据清单中的 best.dataset_path 人工启动。")


if __name__ == "__main__":
    main()
