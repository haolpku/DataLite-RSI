"""Review-only scorer with sample, full-file, and optional dataset-distribution evidence.

The default ``four_dimensions`` mode keeps the general text-task rubric. A
task can select ``criteria`` mode when its domain has a more useful rubric in
``task.quality_criteria``. In that mode the LLM returns evidence for every
criterion, and the framework computes the proportion satisfied.

The four sample-level review dimensions are combined into one comparable score:
- 四维 LLM-judge（各 0-1）：correctness / relevance / difficulty / schema。
- 固定权重复合分：``llm_composite = 0.40·correctness + 0.25·relevance
  + 0.15·difficulty + 0.20·schema``。
- embedding quality（启用时）由已配置的归一化 DAS、normalized Vendi 和 NN-cosine p95 构成；
  占比由 ``embedding_quality.review_score_weight`` 配置，剩余权重属于 sample-level 质量分。sample-level 分在
  ``four_dimensions`` 模式是四维复合分，在 ``criteria`` 模式是 quality criteria 满足比例。
- 硬指标（扫全份文件，不花 LLM）：``dup_rate`` / ``null_rate`` / ``parse_valid_rate``
  / ``contamination_rate``。四维模式会将其作为混合质量分的惩罚；criteria 模式将其
  作为独立的确定性通过门槛并保留具体 issues。
  其中 ``dup_rate`` 取「整记录哈希」与「训练字段级」两种口径的较大值——只看整记录会漏掉
  同一道题配不同措辞 CoT 的情形（详见 ``_scan_validation_metrics``）；``contamination_rate``
  是与 benchmark 测试集的 n-gram 重合率，超阈值直接判 ``passed=False``（见 core/decontamination.py）。
- 抽样从「取前 N 行」改为**随机蓄水池采样**（一趟遍历、O(k) 内存、全份等概率）。
- ``passed`` 同时要求 correctness / relevance / schema 达到各自阈值，用于剔除答案错误、
  任务偏离或结构不合格的节点。
- 可选 EmbeddingQualityEvaluator 固定采样候选集并计算 Vendi Score 与近邻 cosine 分布；
  仅在配置高质量代理集时计算 DAS=-MMD。启用后，这些指标先合成为 embedding quality，
  再按 ``embedding_quality.review_score_weight`` 与 sample-level 分合并；原始指标也会随
  ReviewResult 回传 PipelineAgent。

Optional decontamination references are used only for n-gram overlap detection and are
never included in an Agent prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

from rsi.framework.evolution.agents.base import AgentABC
from rsi.framework.evolution.evaluation.embedding_quality import EmbeddingQualityEvaluator
from rsi.framework.evolution.evaluation.decontamination import ContaminationIndex
from rsi.framework.evolution.models import ReviewResult, TaskSpec
from rsi.framework.evolution.prompts import (
    FAILURE_SIGNALS,
    build_criteria_review_prompt,
    build_review_prompt,
)
from rsi.framework.evolution.telemetry.pipeline_usage import (
    write_combined_iteration_usage,
    write_pipeline_usage_summary,
    write_usage_payload,
)

# Review 复合分的固定权重。
DEFAULT_REVIEW_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "relevance": 0.25,
    "difficulty": 0.15,
    "schema": 0.20,
}

DEFAULT_EMBEDDING_SCORE_WEIGHTS: dict[str, float] = {
    "das": 0.40,
    "vendi": 0.30,
    "nearest_neighbor": 0.30,
}


class ReviewAgent(AgentABC):
    name = "review"

    def __init__(
        self,
        serving,
        four_dimension_thresholds: dict[str, float] | None = None,
        sampling_size: int = 80,
        sampling_seed: int | None = None,
        max_retries: int = 2,
        four_dimension_weights: dict[str, float] | None = None,
        validation_enabled: bool = True,
        contamination: ContaminationIndex | None = None,
        max_null_rate: float = 0.5,
        embedding_quality_evaluator: EmbeddingQualityEvaluator | None = None,
        embedding_quality_fields: tuple[str, str] = ("instruction", "output"),
        embedding_review_score_weight: float | None = None,
        embedding_metric_weights: dict[str, float] | None = None,
        contamination_fields: list[str] | None = None,
        mode: str = "four_dimensions",
        criteria_pass_rate: float = 0.6,
    ) -> None:
        super().__init__(serving, max_retries=max_retries)
        if mode not in {"four_dimensions", "criteria"}:
            raise ValueError("review.mode must be 'four_dimensions' or 'criteria'")
        self.mode = mode
        self.criteria_pass_rate = _clip01(criteria_pass_rate)
        self.four_dimension_thresholds = _normalize_four_dimension_thresholds(
            four_dimension_thresholds
        )
        self.sampling_size = int(sampling_size)
        if self.sampling_size < 1:
            raise ValueError("review.sampling.size 必须至少为 1")
        self.sampling_seed = sampling_seed
        self.four_dimension_weights = _normalize_weights(four_dimension_weights)
        self.validation_enabled = bool(validation_enabled)
        self.max_null_rate = _clip01(max_null_rate)
        # 未配置基准参考集时是一个 disabled 的空索引，扫描恒返回干净——调用点无需分支。
        self.contamination = contamination or ContaminationIndex()
        self.embedding_quality_evaluator = embedding_quality_evaluator
        self.embedding_quality_fields = embedding_quality_fields
        if embedding_quality_evaluator is not None and embedding_review_score_weight is None:
            raise ValueError("启用 embedding_quality 时必须设置 review.embedding_quality.review_score_weight")
        if embedding_review_score_weight is not None and (
            not math.isfinite(float(embedding_review_score_weight))
            or not 0.0 <= float(embedding_review_score_weight) <= 1.0
        ):
            raise ValueError("review.embedding_quality.review_score_weight 必须在 [0, 1] 内")
        self.embedding_review_score_weight = (
            float(embedding_review_score_weight) if embedding_review_score_weight is not None else 0.0
        )
        self.embedding_metric_weights = _normalize_embedding_metric_weights(
            embedding_metric_weights
        )
        self.contamination_fields = (
            list(contamination_fields) if contamination_fields else None
        )

    def build_prompt(self, task: TaskSpec, samples_text: str) -> str:
        if self.mode == "criteria":
            return build_criteria_review_prompt(task, samples_text)
        return build_review_prompt(task, samples_text)

    def compute_composite(self, result: ReviewResult) -> float:
        """Compute the LLM-derived score for the configured review mode."""
        if self.mode == "criteria":
            return _clip01(result.llm_composite)
        w = self.four_dimension_weights
        return (
            w["correctness"] * result.correctness_score
            + w["relevance"] * result.relevance_score
            + w["difficulty"] * result.difficulty_score
            + w["schema"] * result.schema_score
        )

    def parse(self, raw: dict) -> ReviewResult:
        """Parse the LLM response before deterministic full-file checks."""
        if self.mode == "criteria":
            return self._parse_criteria(raw)
        schema_score = _clip01(raw.get("schema_score", 0.0))
        relevance_score = _clip01(raw.get("relevance_score", 0.0))
        # 缺省回退保证部分返回不崩。
        correctness_score = _clip01(raw.get("correctness_score", 0.0))
        difficulty_score = _clip01(raw.get("difficulty_score", 0.0))
        passed = (
            correctness_score >= self.four_dimension_thresholds["correctness"]
            and schema_score >= self.four_dimension_thresholds["schema"]
            and relevance_score >= self.four_dimension_thresholds["relevance"]
        )
        # ANDES F1-F7 失败信号：只规整为合法编号并去重，**不并入 issues**。
        # 两者是异质数据——issues 是针对本份数据的具体证据，信号是标准化标签。混进一个列表后，
        # 每个消费点都得再把它们过滤出来（曾为此写了共享过滤函数 + 属性 + 四处调用）；
        # 保持分离后，需要信号文本的渲染点自己查 FAILURE_SIGNALS 拼即可。
        signals = _normalize_signals(raw.get("failure_signals", []))
        issues = list(raw.get("issues", []))
        result = ReviewResult(
            schema_score=schema_score,
            relevance_score=relevance_score,
            passed=passed,
            issues=issues,
            correctness_score=correctness_score,
            difficulty_score=difficulty_score,
            failure_signals=signals,
        )
        result.llm_composite = self.compute_composite(result)
        # 未叠加硬指标前，review_score 先等于复合分（review() 会用硬指标覆盖）。
        self._update_review_score(result)
        return result

    def _parse_criteria(self, raw: dict) -> ReviewResult:
        if not isinstance(raw, dict):
            raise ValueError("criteria review response must be a JSON object")
        assessments = raw.get("criteria_assessments")
        if not isinstance(assessments, list) or not assessments:
            raise ValueError("criteria_assessments must be a non-empty list")
        indices: set[int] = set()
        clean_assessments: list[dict[str, object]] = []
        for item in assessments:
            if not isinstance(item, dict):
                raise ValueError("criteria_assessments entries must be objects")
            index = item.get("criterion_index")
            met = item.get("met")
            if isinstance(index, bool) or not isinstance(index, int) or index < 1 or index in indices:
                raise ValueError("criterion_index must be a unique positive integer")
            if not isinstance(met, bool):
                raise ValueError("criteria_assessments.met must be boolean")
            indices.add(index)
            clean_assessments.append({
                "criterion_index": index,
                "met": met,
                "evidence": str(item.get("evidence") or "")[:1500],
                "suggestion": str(item.get("suggestion") or "")[:1000],
            })
        critical = raw.get("critical_failures", [])
        if not isinstance(critical, list) or not all(isinstance(item, str) for item in critical):
            raise ValueError("critical_failures must be a list of strings")
        failures = [item.strip()[:1000] for item in critical if item.strip()]
        # In criteria mode the rubric is authoritative: do not ask the model
        # for a second, unconstrained total score. The comparable scalar is
        # the proportion of covered criteria marked as met.
        score = sum(item["met"] for item in clean_assessments) / len(clean_assessments)
        result = ReviewResult(
            schema_score=0.0,
            relevance_score=0.0,
            passed=score >= self.criteria_pass_rate and not failures,
            issues=list(failures),
            llm_composite=score,
            domain_feedback={
                "review_mode": "criteria",
                "criteria_met_rate": score,
                "criteria_assessments": clean_assessments,
                "critical_failures": failures,
            },
        )
        # The overall score is a convenience scalar for incumbent comparison;
        # criterion coverage and evidence remain the authoritative feedback.
        self._update_review_score(result)
        return result

    def apply_validation(
        self, result: ReviewResult, dataset_path: str, task: TaskSpec
    ) -> ReviewResult:
        """扫全份文件得硬指标并叠加惩罚，写回 result 并返回。"""
        required = list(task.target_schema.keys()) if task.target_schema else []
        metrics = _scan_validation_metrics(
            dataset_path,
            required,
            duplicate_fields=None,
        )
        result.dup_rate = metrics.dup_rate
        result.null_rate = metrics.null_rate
        result.parse_valid_rate = metrics.parse_valid_rate
        if metrics.field_dup_rate > metrics.record_dup_rate:
            self.logger.info(
                "重复率由字段 %s 主导：字段级 %.3f > 整记录 %.3f"
                "（同一内容配不同措辞时整记录哈希会失效）",
                metrics.dup_field,
                metrics.field_dup_rate,
                metrics.record_dup_rate,
            )
        if metrics.total == 0:
            # An empty output cannot be selected regardless of sampled LLM scores.
            result.passed = False
            result.review_score = 0.0
            return result

        # Fail fast on a dataset that does not satisfy the configured output contract.
        if required and metrics.null_rate > self.max_null_rate:
            result.passed = False
            result.issues.append(
                f"产出字段与目标 schema 不符：要求至少包含 {required}，"
                f"实测关键字段空缺率 {metrics.null_rate:.1%}（上限 {self.max_null_rate:.1%}）。"
                f"请让流水线最终产出这些字段名且非空（可额外保留中间/溯源字段），"
                f"必需字段名不同会导致后续消费者无法读取。"
            )

        # 去污染：与 benchmark 测试集重合的数据不是"质量差"，是把评测答案抄进了训练集。
        # 既按比例打折（少量疑似重合可能是假阳，不该一票否决），又设硬门槛（超过阈值就是
        # 在刷分，必须判不通过并把证据喂回 Agent，否则搜索会持续往这个方向走）。
        contam = self.contamination.scan(
            dataset_path,
            fields=self.contamination_fields,
        )
        result.contamination_rate = contam.rate
        if contam.hits:
            self.logger.warning(
                "数据与基准测试集重合 %d/%d（%.2f%%）", contam.hits, contam.total, contam.rate * 100
            )
            if contam.rate > self.contamination.max_rate:
                result.passed = False
                result.issues.append(
                    f"检出与评测基准测试集重合 {contam.rate:.1%}（上限 "
                    f"{self.contamination.max_rate:.1%}）：这是数据泄露。命中片段示例："
                    f"{'；'.join(contam.examples)}。请去掉直接取自基准测试集的数据来源。"
                )

        self._update_review_score(result)
        return result

    def review(
        self,
        dataset_path: str,
        task: TaskSpec,
        evidence_path: str | None = None,
        phase: str = "search",
        parent_review: ReviewResult | None = None,
        usage_dir: str | None = None,
    ) -> ReviewResult:
        """LLM 抽样、全量硬指标和可选 embedding 指标共同生成 ReviewResult。

        `evidence_path` appends the exact sample, raw response, and parsed result so every
        feedback step can be audited or rescored offline.
        """
        usage_root = Path(usage_dir).resolve() if usage_dir else None
        review_events = (
            usage_root / "sessions/review/review_llm_calls.jsonl"
            if usage_root is not None
            else None
        )
        if self.mode == "criteria" and not task.quality_criteria:
            result = ReviewResult(
                0.0,
                0.0,
                False,
                ["criteria review requires at least one task.quality_criteria entry"],
            )
            self._dump_evidence(evidence_path, phase, dataset_path, "", None, result)
            self._finalize_usage(usage_root, review_events, result)
            return result

        samples_text = _sample_jsonl(dataset_path, self.sampling_size, seed=self.sampling_seed)
        if not samples_text:
            result = ReviewResult(0.0, 0.0, False, ["数据集为空或不可读"])
            self._dump_evidence(evidence_path, phase, dataset_path, "", None, result)
            self._finalize_usage(usage_root, review_events, result)
            return result
        prompt = self.build_prompt(task, samples_text)
        capture = getattr(self.serving, "capture_usage", None)
        context = (
            capture(review_events, phase=phase, kind="review_llm")
            if callable(capture) and review_events is not None
            else nullcontext()
        )
        try:
            with context:
                raw = self.run_json(prompt)
        finally:
            if usage_root is not None and review_events is not None:
                write_pipeline_usage_summary(
                    review_events,
                    usage_root / "review_llm_token_usage.json",
                )
        if raw is None:
            result = ReviewResult(0.0, 0.0, False, ["评审 LLM 未返回有效 JSON"])
        else:
            try:
                result = self.parse(raw)
                if self.mode == "criteria":
                    expected = set(range(1, len(task.quality_criteria) + 1))
                    actual = {
                        item["criterion_index"]
                        for item in result.domain_feedback["criteria_assessments"]
                    }
                    if actual != expected:
                        result.passed = False
                        result.issues.append(
                            "评审 LLM 未逐条覆盖 task.quality_criteria"
                        )
            except (TypeError, ValueError, KeyError) as exc:
                result = ReviewResult(
                    0.0, 0.0, False,
                    [f"评审 LLM 返回的质量标准反馈无效：{exc}"],
                )
        if self.validation_enabled:
            self.apply_validation(result, dataset_path, task)
        self.apply_embedding_quality(result, dataset_path, parent_review)
        if self.mode == "criteria":
            self.logger.info(
                "Review criteria：quality=%.3f | dup=%.2f null=%.2f parse=%.2f "
                "contamination=%.2f | review_score=%.3f passed=%s",
                result.llm_composite,
                result.dup_rate,
                result.null_rate,
                result.parse_valid_rate,
                result.contamination_rate,
                result.review_score,
                result.passed,
            )
        else:
            self.logger.info(
                "Review sample：corr=%.2f rel=%.2f diff=%.2f schema=%.2f "
                "| dup=%.2f null=%.2f parse=%.2f | composite=%.3f review_score=%.3f passed=%s | signals=%s",
                result.correctness_score, result.relevance_score, result.difficulty_score,
                result.schema_score,
                result.dup_rate, result.null_rate, result.parse_valid_rate,
                result.llm_composite, result.review_score, result.passed,
                ",".join(result.failure_signals) or "-",
            )
        dq = result.embedding_quality
        if dq.enabled:
            self.logger.info(
                "Embedding quality：metrics=%s status=%s MMD=%s DAS=%s delta_vs_parent=%s "
                "Vendi=%s NN-cos-mean=%s samples=%d/%d proxy=%s",
                ",".join(dq.enabled_metrics),
                dq.status,
                f"{dq.mmd:.6f}" if dq.mmd is not None else "-",
                f"{dq.das:.6f}" if dq.das is not None else "-",
                (
                    f"{dq.delta_mmd_vs_parent:+.6f}"
                    if dq.delta_mmd_vs_parent is not None
                    else "-"
                ),
                _format_metric(dq.embedding_diversity, "cosine_vendi_score", 3),
                _format_metric(dq.embedding_diversity, "nearest_neighbor_cosine_mean", 4),
                dq.candidate_sample_size,
                dq.candidate_total,
                dq.proxy_name,
            )
        self._dump_evidence(evidence_path, phase, dataset_path, samples_text, raw, result)
        self._finalize_usage(usage_root, review_events, result)
        return result

    def _finalize_usage(
        self,
        usage_root: Path | None,
        review_events: Path | None,
        result: ReviewResult,
    ) -> None:
        if usage_root is None:
            return
        if review_events is not None:
            write_pipeline_usage_summary(
                review_events,
                usage_root / "review_llm_token_usage.json",
            )
        if result.embedding_quality.enabled:
            write_usage_payload(
                usage_root / "embedding_token_usage.json",
                result.embedding_quality.embedding_usage,
            )
        write_combined_iteration_usage(usage_root)

    def apply_embedding_quality(
        self,
        result: ReviewResult,
        dataset_path: str,
        parent_review: ReviewResult | None = None,
    ) -> ReviewResult:
        """Run optional embedding evaluation and update the comparable score.

        ``criteria`` replaces only the sample-level four-dimension rubric. It
        still uses the same optional dataset-level embedding evidence and
        configured ``embedding_review_score_weight`` as ``four_dimensions`` mode.
        """
        if self.embedding_quality_evaluator is None:
            self._update_review_score(result)
            return result
        user_field, assistant_field = self.embedding_quality_fields
        result.embedding_quality = self.embedding_quality_evaluator.evaluate(
            dataset_path,
            candidate_user_field=user_field,
            candidate_assistant_field=assistant_field,
        )
        parent_quality = parent_review.embedding_quality if parent_review is not None else None
        if (
            result.embedding_quality.mmd is not None
            and parent_quality is not None
            and parent_quality.mmd is not None
        ):
            result.embedding_quality.delta_mmd_vs_parent = (
                result.embedding_quality.mmd - parent_quality.mmd
            )
        if parent_quality is not None:
            result.embedding_quality.delta_embedding_diversity_vs_parent = {
                key: value - parent_quality.embedding_diversity[key]
                for key, value in result.embedding_quality.embedding_diversity.items()
                if key in parent_quality.embedding_diversity
            }
        self._update_review_score(result)
        return result

    def _update_review_score(self, result: ReviewResult) -> None:
        """Compute the comparable score for the selected review mode.

        Criteria mode changes the sample-level component to the proportion of
        satisfied task criteria. If embedding quality evaluation is enabled, that
        component is blended with embedding quality using the same configured
        weight as the default four-dimension mode.
        """
        embedding_score: float | None = None
        embedding_quality_enabled = self.embedding_quality_evaluator is not None
        embedding_review_score_weight = self.embedding_review_score_weight
        if embedding_quality_enabled:
            embedding_score = self._compute_embedding_quality(result)

        if embedding_quality_enabled:
            # Missing enabled evidence receives no embedding credit. This prevents a
            # failed metric computation from outranking candidates with valid evidence.
            dataset_component = embedding_score if embedding_score is not None else 0.0
            combined = (
                (1.0 - embedding_review_score_weight) * result.llm_composite
                + embedding_review_score_weight * dataset_component
            )
        else:
            dataset_component = 0.0
            combined = result.llm_composite

        hard_factor = (
            (1.0 - result.dup_rate)
            * (1.0 - result.null_rate)
            * (1.0 - result.contamination_rate)
            * result.parse_valid_rate
        )
        result.score_components = {
            "llm_composite": result.llm_composite,
            "embedding_quality": dataset_component,
            "embedding_weight": embedding_review_score_weight if embedding_quality_enabled else 0.0,
            "hard_factor": hard_factor,
            "combined_before_hard_penalty": combined,
        }
        # Criteria mode keeps deterministic full-file checks as gates/issues;
        # its comparable score is the sample/embedding blend. The default
        # four-dimension mode retains the historical hard-metric penalty.
        result.review_score = _clip01(
            combined if self.mode == "criteria" else combined * hard_factor
        )

    def _compute_embedding_quality(self, result: ReviewResult) -> float | None:
        quality = result.embedding_quality
        if quality.status != "ok":
            quality.embedding_quality_components = {}
            quality.embedding_quality_score = None
            return None

        diversity = quality.embedding_diversity
        enabled = set(quality.enabled_metrics)
        vendi_ratio = diversity.get("cosine_vendi_ratio")
        nn_p95 = diversity.get("nearest_neighbor_cosine_p95")
        components: dict[str, float] = {}
        if "das" in enabled and quality.mmd is not None:
            # RBF kernels are bounded in [0, 1], so MMD is bounded by sqrt(2).
            components["das"] = _clip01(1.0 - quality.mmd / math.sqrt(2.0))
        if "vendi" in enabled and vendi_ratio is not None:
            components["vendi"] = _clip01(vendi_ratio)
        if "nearest_neighbor" in enabled and nn_p95 is not None:
            components["nearest_neighbor"] = _clip01((1.0 - nn_p95) / 2.0)

        weights = getattr(
            self, "embedding_metric_weights", DEFAULT_EMBEDDING_SCORE_WEIGHTS
        )
        available_weight = sum(weights[key] for key in components)
        if available_weight <= 0.0:
            return None
        score = sum(
            weights[key] * value
            for key, value in components.items()
        ) / available_weight
        quality.embedding_quality_components = components
        quality.embedding_quality_score = _clip01(score)
        return quality.embedding_quality_score

    def _dump_evidence(
        self,
        evidence_path: str | None,
        phase: str,
        dataset_path: str,
        samples_text: str,
        parsed: dict | None,
        result: ReviewResult,
    ) -> None:
        """把一次评审的完整证据追加为一行 JSON。落盘失败绝不影响主流程。

        Append instead of overwrite so repeated review calls remain auditable.
        """
        if not evidence_path:
            return
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "dataset_path": dataset_path,
            "sampling_size": self.sampling_size,
            "seed": self.sampling_seed,
            "weights": self.four_dimension_weights,
            "samples_jsonl": samples_text,     # 抽样原文：LLM 实际看到的字节，本身即合法 JSONL
            "raw_response": self.last_raw_response,
            "parsed": parsed,                  # 抽取后的 JSON（None = 未返回有效 JSON）
            "result": asdict(result),
        }
        try:
            path = Path(evidence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self.logger.warning("评审证据落盘失败（不影响主流程）：%s", exc)


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """合并用户权重与默认权重（缺项回退默认）；不强制归一，保持固定权重语义。"""
    merged = dict(DEFAULT_REVIEW_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in merged:
                merged[k] = float(v)
    return merged


def _normalize_four_dimension_thresholds(
    thresholds: dict[str, float] | None,
) -> dict[str, float]:
    merged = {"correctness": 0.6, "relevance": 0.6, "schema": 0.6}
    if thresholds:
        for key, value in thresholds.items():
            if key not in merged:
                raise ValueError(f"review.four_dimensions.pass_thresholds 未知字段：{key}")
            merged[key] = _clip01(value)
    return merged


def _normalize_embedding_metric_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    merged = dict(DEFAULT_EMBEDDING_SCORE_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key in merged:
                merged[key] = max(0.0, float(value))
    if sum(merged.values()) <= 0.0:
        raise ValueError("embedding_quality.metrics 至少需要一个启用且有正权重的指标")
    return merged


def _format_metric(metrics: dict[str, float], key: str, precision: int) -> str:
    value = metrics.get(key)
    return f"{value:.{precision}f}" if value is not None else "-"


def _normalize_signals(raw_signals) -> list[str]:
    """把 LLM 返回的 failure_signals 规整为合法编号（F1-F7）、大写、去重、保序。"""
    if not isinstance(raw_signals, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw_signals:
        sid = str(x).strip().upper()
        if sid in FAILURE_SIGNALS and sid not in out:
            out.append(sid)
    return out


def _clip01(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _sample_jsonl(path: str, n: int, seed: int | None = None) -> str:
    """随机蓄水池采样：一趟遍历、O(n) 内存、全份非空行等概率抽 n 条。

    相比旧的「取前 N 行」，蓄水池采样避免只看到数据集开头（生成类算子常按类别/难度
    顺序产出，前 N 行有偏），使 Review 打分更能代表整份数据。
    """
    p = Path(path)
    if not p.exists():
        return ""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = line
    return "\n".join(reservoir)


class HardMetrics(NamedTuple):
    """``_scan_validation_metrics`` 的返回值。dup_rate 是对外口径，另两个 dup 仅供日志溯源。"""

    dup_rate: float
    record_dup_rate: float
    field_dup_rate: float
    dup_field: str
    null_rate: float
    parse_valid_rate: float
    total: int


# 内容字段判定门槛：中位长度低于此值的一律当标签字段看待。answer 常常只是一个数字、
# difficulty_level 只有 easy/medium/hard，这类字段天然大量重复，算进重复率会把每份数据集
# 都判成重灾区。40 字符落在标签字段（个位数长度）和题干/推理链（数百字符）之间的空档里。
CONTENT_FIELD_MIN_MEDIAN_LEN = 40

# 求中位数需要留样本。只留前 N 条：区分「3 字符的答案」和「300 字符的题干」根本不需要
# 无偏样本，而阈值离两种量级都很远，前缀偏置翻不动这个判定。
_LENGTH_SAMPLE_CAP = 4096


def _scan_validation_metrics(
    path: str,
    required_fields: list[str],
    *,
    duplicate_fields: list[str] | None = None,
) -> HardMetrics:
    """扫全份 jsonl 得硬指标。

    - dup_rate: 重复占比，取 ``max(整记录哈希重复率, 训练字段级重复率)``。
      整记录哈希只抓「整行一模一样」，会漏掉同一道题配不同措辞 CoT 的情形——两行哈希不同，
      但对模型而言是同一份监督信号。实测某份 4000 条数据整记录只报 0.06，而 question
      字段唯一率仅 0.518。反过来字段级也补不了整记录：题干各异而整行重复同样是重复。
      两种口径互不覆盖，所以取较大值。
    - null_rate: 关键字段空缺率（字段级）= 空缺槽位 / (可解析行数 × 字段数)；无 required
      字段时为 0。
    - parse_valid_rate: 合法 JSON 对象行 / total。
    """
    p = Path(path)
    if not p.exists():
        return HardMetrics(0.0, 0.0, 0.0, "", 0.0, 1.0, 0)

    total = 0
    parseable = 0
    hashes: set[str] = set()
    null_slots = 0
    field_slots = 0
    # Null/schema coverage always uses every required field. Duplicate evidence may
    # use a narrower set so a deliberately fixed task instruction is not mistaken
    # for collapsed sample content.
    content_fields = duplicate_fields if duplicate_fields is not None else required_fields
    field_hashes: dict[str, set[str]] = {fld: set() for fld in content_fields}
    field_lengths: dict[str, list[int]] = {fld: [] for fld in content_fields}

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = None
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                obj = None

            if isinstance(obj, dict):
                parseable += 1
                canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                hashes.add(hashlib.md5(canonical.encode("utf-8")).hexdigest())
                for fld in required_fields:
                    field_slots += 1
                    value = obj.get(fld)
                    if _is_empty(value):
                        null_slots += 1
                for fld in content_fields:
                    value = obj.get(fld)
                    text = _as_text(value)
                    field_hashes[fld].add(
                        hashlib.md5(text.encode("utf-8")).hexdigest()
                    )
                    if len(field_lengths[fld]) < _LENGTH_SAMPLE_CAP:
                        field_lengths[fld].append(len(text))
            else:
                hashes.add(hashlib.md5(line.encode("utf-8")).hexdigest())

    if total == 0:
        return HardMetrics(0.0, 0.0, 0.0, "", 0.0, 0.0, 0)

    record_dup = _clip01(1.0 - (len(hashes) / total))
    field_dup, dup_field = _worst_content_field_dup(
        field_hashes, field_lengths, parseable
    )
    parse_valid_rate = parseable / total
    null_rate = (null_slots / field_slots) if field_slots > 0 else 0.0
    return HardMetrics(
        dup_rate=max(record_dup, field_dup),
        record_dup_rate=record_dup,
        field_dup_rate=field_dup,
        dup_field=dup_field,
        null_rate=_clip01(null_rate),
        parse_valid_rate=_clip01(parse_valid_rate),
        total=total,
    )


def _worst_content_field_dup(
    field_hashes: dict[str, set[str]],
    field_lengths: dict[str, list[int]],
    rows: int,
) -> tuple[float, str]:
    """训练字段里塌陷最严重那一维的重复率，返回 (dup_rate, 字段名)。

    只看内容字段（中位长度 ≥ CONTENT_FIELD_MIN_MEDIAN_LEN），标签字段一律跳过。
    """
    if rows <= 0:
        return 0.0, ""
    worst = 0.0
    worst_field = ""
    for fld, hashes in field_hashes.items():
        lengths = field_lengths.get(fld) or []
        if not lengths or statistics.median(lengths) < CONTENT_FIELD_MIN_MEDIAN_LEN:
            continue
        dup = 1.0 - (len(hashes) / rows)
        if dup > worst:
            worst, worst_field = dup, fld
    return _clip01(worst), worst_field


def _as_text(value) -> str:
    """把字段值折成可哈希、可量长度的字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False
