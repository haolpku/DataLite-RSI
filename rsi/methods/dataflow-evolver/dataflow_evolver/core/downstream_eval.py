"""Process-isolated downstream SFT and benchmark evaluation feedback.

The core package intentionally knows nothing about a training framework.  It writes
one JSON request, invokes a configured command, and validates the JSON result that
the external environment writes back.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dataflow_evolver.core.schemas import TaskSpec


@dataclass(frozen=True)
class DownstreamEvaluationRequest:
    """Inputs handed to an external SFT/evaluation runner."""

    checkpoint_iteration: int
    incumbent_iteration: int | None
    dataset_path: str | None
    pipeline_path: str | None
    task: TaskSpec
    expected_benchmarks: list[str]
    bad_cases_per_benchmark: int
    stage: str = "periodic"
    training_performed: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["protocol_version"] = 3
        return payload


@dataclass(frozen=True)
class DownstreamBadCase:
    question: str
    model_response: str
    parsed_answer: str
    correct_answer: str
    response_token_count: int | None = None
    finish_reason: str = ""
    length_truncated: bool = False
    dimension: str = ""


@dataclass(frozen=True)
class DownstreamBenchmarkResult:
    score: float
    metric: str
    incorrect_count: int
    wrong_answer_count: int
    parse_failure_count: int
    runaway_count: int
    bad_cases: list[DownstreamBadCase]
    correct_count: int | None = None
    total_count: int | None = None
    question_count: int | None = None
    n_sampling: int | None = None


@dataclass
class DownstreamAttributionResult:
    """Structured, data-side diagnosis of downstream benchmark failures.

    The raw bad cases remain audit-only evidence. This report is the
    deliberately smaller contract that may be shown to the next
    PipelineAgent.
    """

    status: str = "failed"
    summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    non_data_causes: list[str] = field(default_factory=list)
    recommended_pipeline_changes: list[str] = field(default_factory=list)
    error: str | None = None

@dataclass
class DownstreamEvaluationResult:
    """Validated feedback, including infrastructure failures as explicit records."""

    status: str
    checkpoint_iteration: int
    incumbent_iteration: int | None
    dataset_path: str | None
    stage: str = "periodic"
    training_performed: bool = True
    model: str = ""
    benchmarks: dict[str, DownstreamBenchmarkResult] = field(default_factory=dict)
    score_deltas: dict[str, float] = field(default_factory=dict)
    score_deltas_vs_baseline: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    request_path: str = ""
    result_path: str = ""
    error: str | None = None
    attribution: DownstreamAttributionResult | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class DownstreamEvaluatorABC(ABC):
    """Boundary implemented by process-backed or test evaluators."""

    expected_benchmarks: list[str]
    bad_cases_per_benchmark: int
    baseline_enabled: bool = False

    @abstractmethod
    def run(
        self,
        request: DownstreamEvaluationRequest,
        checkpoint_dir: Path,
    ) -> DownstreamEvaluationResult:
        raise NotImplementedError


class CommandDownstreamEvaluator(DownstreamEvaluatorABC):
    """Invoke an external runner without importing its training dependencies.

    The configured command is treated as an argv prefix.  The framework appends
    ``--request <path> --result <path>`` and never invokes a shell.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        expected_benchmarks: Sequence[str],
        bad_cases_per_benchmark: int = 3,
        timeout_sec: int = 86400,
        baseline_enabled: bool = False,
    ) -> None:
        self.command = [str(part) for part in command if str(part)]
        self.expected_benchmarks = [str(name) for name in expected_benchmarks if str(name)]
        self.bad_cases_per_benchmark = int(bad_cases_per_benchmark)
        self.timeout_sec = int(timeout_sec)
        self.baseline_enabled = bool(baseline_enabled)
        if not self.command:
            raise ValueError("downstream_eval.command must contain at least one argv item")
        if not self.expected_benchmarks:
            raise ValueError("downstream_eval.benchmarks must not be empty")
        if self.bad_cases_per_benchmark < 1:
            raise ValueError("downstream_eval.bad_cases_per_benchmark must be at least 1")
        if self.timeout_sec < 1:
            raise ValueError("downstream_eval.timeout_sec must be at least 1")

    def run(
        self,
        request: DownstreamEvaluationRequest,
        checkpoint_dir: Path,
    ) -> DownstreamEvaluationResult:
        checkpoint_dir = checkpoint_dir.resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        request_path = checkpoint_dir / "request.json"
        result_path = checkpoint_dir / "result.json"
        stdout_path = checkpoint_dir / "stdout.log"
        stderr_path = checkpoint_dir / "stderr.log"
        request_path.write_text(
            json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        argv = [
            *self.command,
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        started = time.perf_counter()
        try:
            if request.stage == "baseline":
                from os import environ

                cache_path_value = environ.get(
                    "DF_DOWNSTREAM_BASELINE_RESULT_PATH", ""
                ).strip()
                if cache_path_value:
                    cache_path = Path(cache_path_value).expanduser().resolve()
                    if not cache_path.is_file():
                        raise FileNotFoundError(
                            f"baseline cache result does not exist: {cache_path}"
                        )
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    benchmarks = _validate_result(
                        raw,
                        expected_benchmarks=self.expected_benchmarks,
                        bad_cases_per_benchmark=self.bad_cases_per_benchmark,
                    )
                    result_path.write_bytes(cache_path.read_bytes())
                    stdout_path.write_text(
                        f"Reused baseline result from {cache_path}\n",
                        encoding="utf-8",
                    )
                    stderr_path.write_text("", encoding="utf-8")
                    return DownstreamEvaluationResult(
                        status="ok",
                        checkpoint_iteration=request.checkpoint_iteration,
                        incumbent_iteration=request.incumbent_iteration,
                        dataset_path=request.dataset_path,
                        stage=request.stage,
                        training_performed=request.training_performed,
                        model=_nonempty_string(raw.get("model"), "model"),
                        benchmarks=benchmarks,
                        summary=str(raw.get("summary", "") or ""),
                        artifacts=_string_mapping(raw.get("artifacts", {}), "artifacts"),
                        wall_time_seconds=round(time.perf_counter() - started, 6),
                        request_path=str(request_path),
                        result_path=str(result_path),
                    )

            completed = subprocess.run(
                argv,
                cwd=str(checkpoint_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_sec,
                check=False,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode != 0:
                return self._failure(
                    request,
                    request_path,
                    result_path,
                    started,
                    f"external runner exited with code {completed.returncode}",
                )
            if not result_path.is_file():
                return self._failure(
                    request,
                    request_path,
                    result_path,
                    started,
                    "external runner did not create result.json",
                )
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            benchmarks = _validate_result(
                raw,
                expected_benchmarks=self.expected_benchmarks,
                bad_cases_per_benchmark=self.bad_cases_per_benchmark,
            )
            return DownstreamEvaluationResult(
                status="ok",
                checkpoint_iteration=request.checkpoint_iteration,
                incumbent_iteration=request.incumbent_iteration,
                dataset_path=request.dataset_path,
                stage=request.stage,
                training_performed=request.training_performed,
                model=_nonempty_string(raw.get("model"), "model"),
                benchmarks=benchmarks,
                summary=str(raw.get("summary", "") or ""),
                artifacts=_string_mapping(raw.get("artifacts", {}), "artifacts"),
                wall_time_seconds=round(time.perf_counter() - started, 6),
                request_path=str(request_path),
                result_path=str(result_path),
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(_process_text(exc.stdout), encoding="utf-8")
            stderr_path.write_text(_process_text(exc.stderr), encoding="utf-8")
            return self._failure(
                request,
                request_path,
                result_path,
                started,
                f"external runner timed out after {self.timeout_sec}s",
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return self._failure(
                request,
                request_path,
                result_path,
                started,
                f"invalid downstream evaluation: {exc}",
            )

    @staticmethod
    def _failure(
        request: DownstreamEvaluationRequest,
        request_path: Path,
        result_path: Path,
        started: float,
        error: str,
    ) -> DownstreamEvaluationResult:
        return DownstreamEvaluationResult(
            status="failed",
            checkpoint_iteration=request.checkpoint_iteration,
            incumbent_iteration=request.incumbent_iteration,
            dataset_path=request.dataset_path,
            stage=request.stage,
            training_performed=request.training_performed,
            wall_time_seconds=round(time.perf_counter() - started, 6),
            request_path=str(request_path),
            result_path=str(result_path),
            error=error,
        )


def result_to_dict(result: DownstreamEvaluationResult) -> dict[str, Any]:
    """Serialize nested dataclasses for traces, manifests, and prompt tests."""
    return asdict(result)


def _validate_result(
    raw: Any,
    *,
    expected_benchmarks: Sequence[str],
    bad_cases_per_benchmark: int,
) -> dict[str, DownstreamBenchmarkResult]:
    if not isinstance(raw, dict):
        raise ValueError("result root must be a JSON object")
    raw_benchmarks = raw.get("benchmarks")
    if not isinstance(raw_benchmarks, dict):
        raise ValueError("benchmarks must be a JSON object")
    missing = [name for name in expected_benchmarks if name not in raw_benchmarks]
    if missing:
        raise ValueError(f"missing benchmark results: {', '.join(missing)}")

    validated: dict[str, DownstreamBenchmarkResult] = {}
    for name in expected_benchmarks:
        item = raw_benchmarks[name]
        if not isinstance(item, dict):
            raise ValueError(f"benchmark {name!r} must be a JSON object")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"benchmark {name!r} score must be numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"benchmark {name!r} score must be finite")
        incorrect_count = _nonnegative_int(
            item.get("incorrect_count"), f"{name}.incorrect_count"
        )
        count_fields = (
            "wrong_answer_count",
            "parse_failure_count",
            "runaway_count",
        )
        present_count_fields = [field in item for field in count_fields]
        if any(present_count_fields) and not all(present_count_fields):
            raise ValueError(
                f"benchmark {name!r} must provide all bad-case category counts"
            )
        if all(present_count_fields):
            category_counts = {
                field: _nonnegative_int(item.get(field), f"{name}.{field}")
                for field in count_fields
            }
            if sum(category_counts.values()) != incorrect_count:
                raise ValueError(
                    f"benchmark {name!r} bad-case category counts must sum to "
                    f"incorrect_count={incorrect_count}"
                )
        else:
            category_counts = {
                "wrong_answer_count": incorrect_count,
                "parse_failure_count": 0,
                "runaway_count": 0,
            }
        expected_bad_cases = min(bad_cases_per_benchmark, incorrect_count)
        bad_cases = item.get("bad_cases")
        if not isinstance(bad_cases, list) or len(bad_cases) != expected_bad_cases:
            raise ValueError(
                f"benchmark {name!r} must contain exactly {expected_bad_cases} bad_cases "
                f"(min({bad_cases_per_benchmark}, incorrect_count={incorrect_count}))"
            )
        metric = _nonempty_string(item.get("metric"), f"{name}.metric")
        result_count_fields = (
            "correct_count",
            "total_count",
            "question_count",
            "n_sampling",
        )
        present_result_counts = [field in item for field in result_count_fields]
        if any(present_result_counts) and not all(present_result_counts):
            raise ValueError(
                f"benchmark {name!r} must provide all result count fields"
            )
        if all(present_result_counts):
            correct_count = _nonnegative_int(
                item.get("correct_count"), f"{name}.correct_count"
            )
            total_count = _nonnegative_int(
                item.get("total_count"), f"{name}.total_count"
            )
            question_count = _nonnegative_int(
                item.get("question_count"), f"{name}.question_count"
            )
            n_sampling = _nonnegative_int(
                item.get("n_sampling"), f"{name}.n_sampling"
            )
            if total_count < 1 or question_count < 1 or n_sampling < 1:
                raise ValueError(
                    f"benchmark {name!r} result counts must be positive except correct_count"
                )
            if correct_count > total_count:
                raise ValueError(
                    f"benchmark {name!r} correct_count exceeds total_count"
                )
            if total_count - correct_count != incorrect_count:
                raise ValueError(
                    f"benchmark {name!r} count mismatch: total-correct="
                    f"{total_count - correct_count}, incorrect_count={incorrect_count}"
                )
            expected_total = (
                question_count * n_sampling
                if metric == f"avg@{n_sampling}"
                else question_count
            )
            if total_count != expected_total:
                raise ValueError(
                    f"benchmark {name!r} total_count={total_count} does not match "
                    f"metric={metric}, question_count={question_count}, "
                    f"n_sampling={n_sampling}"
                )
            if abs(score - correct_count / total_count) > 1e-9:
                raise ValueError(
                    f"benchmark {name!r} score does not equal correct_count/total_count"
                )
        else:
            # Backward-compatible inference for cached protocol-v3 baseline results.
            n_sampling = 1
            if "@" in metric:
                try:
                    n_sampling = int(metric.rsplit("@", 1)[1])
                except ValueError:
                    n_sampling = 1
            if score < 1.0:
                total_count = round(incorrect_count / (1.0 - score))
                correct_count = total_count - incorrect_count
                question_count = total_count
            else:
                correct_count = None
                total_count = None
                question_count = None
        validated[name] = DownstreamBenchmarkResult(
            score=score,
            metric=metric,
            incorrect_count=incorrect_count,
            wrong_answer_count=category_counts["wrong_answer_count"],
            parse_failure_count=category_counts["parse_failure_count"],
            runaway_count=category_counts["runaway_count"],
            bad_cases=[
                _validate_bad_case(name, index, value)
                for index, value in enumerate(bad_cases)
            ],
            correct_count=correct_count,
            total_count=total_count,
            question_count=question_count,
            n_sampling=n_sampling,
        )
    return validated


def _validate_bad_case(name: str, index: int, raw: Any) -> DownstreamBadCase:
    if not isinstance(raw, dict):
        raise ValueError(f"{name}.bad_cases[{index}] must be a JSON object")
    return DownstreamBadCase(
        question=_nonempty_string(raw.get("question"), f"{name}.bad_cases[{index}].question"),
        model_response=_nonempty_string(
            raw.get("model_response"), f"{name}.bad_cases[{index}].model_response"
        ),
        parsed_answer=_string(
            raw.get("parsed_answer"), f"{name}.bad_cases[{index}].parsed_answer"
        ),
        correct_answer=_nonempty_string(
            raw.get("correct_answer"), f"{name}.bad_cases[{index}].correct_answer"
        ),
        response_token_count=_optional_nonnegative_int(
            raw.get("response_token_count"),
            f"{name}.bad_cases[{index}].response_token_count",
        ),
        finish_reason=_string(
            raw.get("finish_reason", ""), f"{name}.bad_cases[{index}].finish_reason"
        ),
        length_truncated=_bool(
            raw.get("length_truncated", False),
            f"{name}.bad_cases[{index}].length_truncated",
        ),
        dimension=_string(
            raw.get("dimension", ""), f"{name}.bad_cases[{index}].dimension"
        ),
    )


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"{field_name} must map strings to strings")
    return dict(value)


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
