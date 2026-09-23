#!/usr/bin/env python3


import json
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import signal

import httpx
import pyarrow.parquet as parquet
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()


@dataclass
class ModelEndpointConfig:
    """一个 OpenAI-compatible 模型端点。"""

    model: str
    base_url: str
    api_key_env: str
    max_tokens: int = 16000
    temperature: float = 1.0
    timeout: int = 180
    connect_timeout: int = 30
    request_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_model(self) -> str:
        """将 Inspect 的 openai-api/<service>/<model> 转成接口模型名。"""
        parts = self.model.split("/")
        if len(parts) >= 3 and parts[0] == "openai-api":
            return "/".join(parts[2:])
        return self.model


@dataclass
class EvalConfig:
    """评估配置。"""

    model: ModelEndpointConfig
    grader: ModelEndpointConfig
    max_workers: int = 4
    max_samples: Optional[int] = None
    checkpoint_interval: int = 300  # 秒
    retry_attempts: int = 3


@dataclass
class SampleResult:
    """单个样本结果"""
    id: str
    question: str
    correct_answer: str
    model_answer: Optional[str] = None
    grade: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """断点管理器"""
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.last_save = time.time()

    def save(self, results: list[SampleResult], interval: int):
        """保存检查点"""
        if time.time() - self.last_save < interval:
            return

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]

        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.checkpoint_file)

        self.last_save = time.time()
        console.print(f"[dim]Checkpoint saved: {len(results)} results[/dim]")

    def load(self) -> list[Dict]:
        """加载检查点"""
        if not self.checkpoint_file.exists():
            return []

        with open(self.checkpoint_file) as f:
            return json.load(f)


class HLELiteEvaluator:
    """轻量级 HLE 评估器"""

    def __init__(
        self,
        config: EvalConfig,
        data_dir: Path,
        output_dir: Path,
        checkpoint_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
        metrics_file: Optional[Path] = None,
    ):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Solver 和 grader 可以使用完全独立的供应商和凭据。
        self.api_keys = {}
        for role, endpoint in (("solver", config.model), ("grader", config.grader)):
            api_key = os.environ.get(endpoint.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Environment variable {endpoint.api_key_env} not set "
                    f"for {role} model {endpoint.model}"
                )
            self.api_keys[role] = api_key

        # Checkpoint
        self.checkpoint_file = checkpoint_file or output_dir / "checkpoint.json"
        self.results_file = results_file or output_dir / "results.json"
        self.metrics_file = metrics_file or output_dir / "metrics.json"
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_file)

        # State
        self.results: list[SampleResult] = []
        self.should_stop = False

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        console.print("\n[yellow]Received interrupt signal, saving checkpoint...[/yellow]")
        self.should_stop = True
        self.checkpoint_mgr.save(self.results, 0)
        console.print("[green]Checkpoint saved. Exiting.[/green]")
        sys.exit(0)

    def load_samples(self) -> list[SampleResult]:
        """加载 HLE 样本"""
        samples = []

        parquet_files = sorted(self.data_dir.glob("Revision_subset.part*.parquet"))
        if not parquet_files:
            parquet_files = sorted(self.data_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            table = parquet.read_table(parquet_file)
            for row in table.to_pylist():
                raw_record = row.get("json")
                record = json.loads(raw_record) if isinstance(raw_record, str) else row
                samples.append(SampleResult(
                    id=str(record.get("id")),
                    question=record.get("question"),
                    correct_answer=str(record.get("answer")),
                ))

                if self.config.max_samples and len(samples) >= self.config.max_samples:
                    return samples

        return samples

    def call_api(self, role: str, messages: list[Dict]) -> Dict:
        """调用 API"""
        endpoint = self.config.model if role == "solver" else self.config.grader
        headers = {
            "Authorization": f"Bearer {self.api_keys[role]}",
            "Content-Type": "application/json",
        }

        payload = dict(endpoint.request_config)
        payload.update(
            {
                "model": endpoint.api_model,
                "messages": messages,
                "max_tokens": endpoint.max_tokens,
                "temperature": endpoint.temperature,
            }
        )

        for attempt in range(self.config.retry_attempts):
            try:
                timeout = httpx.Timeout(
                    endpoint.timeout,
                    connect=endpoint.connect_timeout,
                )
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{endpoint.base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    return {"error": str(e)}
                time.sleep(2 ** attempt)

        return {"error": "Max retries exceeded"}

    def solve_sample(self, sample: SampleResult) -> SampleResult:
        """求解单个样本"""
        if self.should_stop:
            return sample

        messages = [
            {"role": "user", "content": f"Question: {sample.question}\n\nProvide your answer."}
        ]

        response = self.call_api("solver", messages)

        if "error" in response:
            sample.status = "error"
            sample.error = response["error"]
        else:
            try:
                sample.model_answer = response["choices"][0]["message"]["content"]
                sample.status = "solved"
                sample.timestamp = time.time()
            except (KeyError, IndexError) as e:
                sample.status = "error"
                sample.error = f"Parse error: {e}"

        return sample

    def grade_sample(self, sample: SampleResult) -> SampleResult:
        """评分单个样本"""
        if self.should_stop or sample.status != "solved":
            return sample

        messages = [{
            "role": "user",
            "content": f"""Question: {sample.question}

Correct Answer: {sample.correct_answer}

Student Answer: {sample.model_answer}

Is the student answer correct? Reply with just "CORRECT" or "INCORRECT"."""
        }]

        response = self.call_api("grader", messages)

        if "error" in response:
            sample.grade = "error"
            sample.error = f"Grading error: {response['error']}"
        else:
            try:
                grade_text = response["choices"][0]["message"]["content"].strip().upper()
                sample.grade = "correct" if "CORRECT" in grade_text and "INCORRECT" not in grade_text else "incorrect"
            except (KeyError, IndexError):
                sample.grade = "error"

        return sample

    def run(self):
        """运行评估"""
        console.print("\n[bold cyan]HLE Lite Evaluation[/bold cyan]")
        console.print(f"Model: {self.config.model.model}")
        console.print(f"Grader: {self.config.grader.model}")
        console.print(f"Workers: {self.config.max_workers}")
        console.print(f"Max samples: {self.config.max_samples or 'all'}")
        console.print(f"Checkpoint interval: {self.config.checkpoint_interval}s")
        console.print()

        # 加载样本
        console.print("Loading samples...")
        samples = self.load_samples()
        console.print(f"Loaded {len(samples)} samples\n")

        # 加载 checkpoint（如果存在）
        checkpoint_data = self.checkpoint_mgr.load()
        completed_ids = {r["id"] for r in checkpoint_data if r.get("status") == "solved"}

        if completed_ids:
            console.print(f"[yellow]Found checkpoint with {len(completed_ids)} completed samples[/yellow]\n")
            self.results = [SampleResult(**r) for r in checkpoint_data]
            samples = [s for s in samples if s.id not in completed_ids]

        if not samples:
            console.print("[green]All samples already completed![/green]")
            return

        # 求解
        console.print(f"[bold]Solving {len(samples)} problems...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Solving...", total=len(samples))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self.solve_sample, s): s for s in samples}

                for future in as_completed(futures):
                    if self.should_stop:
                        break

                    result = future.result()
                    self.results.append(result)
                    progress.advance(task)

                    # 定期保存 checkpoint
                    self.checkpoint_mgr.save(self.results, self.config.checkpoint_interval)

        console.print()

        # 评分
        to_grade = [r for r in self.results if r.status == "solved" and not r.grade]
        if to_grade:
            console.print(f"[bold]Grading {len(to_grade)} answers...[/bold]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Grading...", total=len(to_grade))

                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {executor.submit(self.grade_sample, r): r for r in to_grade}

                    for future in as_completed(futures):
                        if self.should_stop:
                            break

                        future.result()
                        progress.advance(task)

                        self.checkpoint_mgr.save(self.results, self.config.checkpoint_interval)

            console.print()

        # 保存最终结果
        self.save_results()
        self.save_metrics()

        # 显示统计
        self.print_stats()

    def save_results(self):
        """保存结果"""
        with open(self.results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        console.print(f"[green]Results saved to: {self.results_file}[/green]")

    def save_metrics(self):
        """保存 benchmark 可读取的结构化汇总指标。"""
        total = len(self.results)
        solved = sum(1 for result in self.results if result.status == "solved")
        graded = sum(
            1 for result in self.results if result.grade in {"correct", "incorrect"}
        )
        correct = sum(1 for result in self.results if result.grade == "correct")
        metrics = {
            "metrics": {"accuracy": correct / graded if graded else None},
            "counts": {
                "total": total,
                "solved": solved,
                "graded": graded,
                "correct": correct,
                "errors": sum(
                    1
                    for result in self.results
                    if result.status == "error" or result.grade == "error"
                ),
            },
        }
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        console.print(f"[green]Metrics saved to: {self.metrics_file}[/green]")

    def print_stats(self):
        """打印统计信息"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.status == "solved")
        correct = sum(1 for r in self.results if r.grade == "correct")

        console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print("[bold cyan]Results Summary[/bold cyan]")
        console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print(f"Total samples: {total}")
        console.print(f"Successfully solved: {success}")
        console.print(f"Correct answers: {correct}")
        if success > 0:
            console.print(f"[bold]Accuracy: {correct/success*100:.1f}%[/bold]")
        console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

def _load_model_endpoint(
    data: Dict[str, Any],
    fallback: Optional[ModelEndpointConfig] = None,
    default_max_tokens: Optional[int] = None,
) -> ModelEndpointConfig:
    """读取 Inspect ModelConfig，同时兼容旧 lite 配置。"""
    is_inspect_config = "model" in data
    generate_config = data.get("config", {}) if is_inspect_config else data
    model_args = data.get("args", {}) if is_inspect_config else {}

    default_model = fallback.model if fallback else "openai-api/zcloud/claude-sonnet-5"
    default_base_url = fallback.base_url if fallback else "https://api.zcloudapi.com/v1"
    default_api_key_env = fallback.api_key_env if fallback else "ZCLOUD_API_KEY"
    if default_max_tokens is None:
        default_max_tokens = fallback.max_tokens if fallback else 16000
    default_temperature = fallback.temperature if fallback else 1.0
    default_timeout = fallback.timeout if fallback else 180
    default_connect_timeout = fallback.connect_timeout if fallback else 30

    known_generate_keys = {"max_tokens", "temperature"}
    request_config = {
        key: value
        for key, value in generate_config.items()
        if key not in known_generate_keys
    }

    model = (
        data.get("model", default_model)
        if is_inspect_config
        else data.get("name", default_model)
    )
    return ModelEndpointConfig(
        model=model,
        base_url=data.get("base_url", default_base_url),
        api_key_env=model_args.get("api_key_var", default_api_key_env)
        if is_inspect_config
        else data.get("api_key_env", default_api_key_env),
        max_tokens=generate_config.get("max_tokens", default_max_tokens),
        temperature=generate_config.get("temperature", default_temperature),
        timeout=model_args.get("client_timeout", default_timeout)
        if is_inspect_config
        else data.get("timeout", default_timeout),
        connect_timeout=model_args.get("client_connect_timeout", default_connect_timeout),
        request_config=request_config,
    )


def load_config_file(config_path: Path) -> EvalConfig:
    """从 YAML 文件加载配置"""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    model_config = _load_model_endpoint(data.get("model", {}))

    # 正式 hle_eval 使用 model_roles.grader；旧 lite 配置使用 judge。
    grader_data = data.get("model_roles", {}).get("grader")
    if grader_data is None:
        grader_data = data.get("judge", {})
    grader_config = _load_model_endpoint(
        grader_data,
        fallback=model_config,
        default_max_tokens=100,
    )

    return EvalConfig(
        model=model_config,
        grader=grader_config,
        max_workers=data.get("max_workers", 4),
        max_samples=data.get("max_samples"),
        checkpoint_interval=data.get("checkpoint_interval", 300),
        retry_attempts=data.get("retry_attempts", 3),
    )


def main():
    parser = argparse.ArgumentParser(description="HLE Lite Evaluation")
    parser.add_argument("--config", type=Path, help="Config YAML file")
    parser.add_argument("--data-dir", type=Path, help="HLE data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--max-workers", type=int, help="Max concurrent workers")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate")
    parser.add_argument("--checkpoint-interval", type=int, help="Checkpoint interval (seconds)")
    parser.add_argument("--checkpoint-file", type=Path, help="Stable checkpoint path")
    parser.add_argument("--results-file", type=Path, help="Per-sample results path")
    parser.add_argument("--metrics-file", type=Path, help="Aggregate metrics path")

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = load_config_file(args.config)
    else:
        config = EvalConfig(
            model=ModelEndpointConfig(
                model="openai-api/zcloud/claude-sonnet-5",
                base_url="https://api.zcloudapi.com/v1",
                api_key_env="ZCLOUD_API_KEY",
            ),
            grader=ModelEndpointConfig(
                model="openai-api/zcloud/claude-sonnet-5",
                base_url="https://api.zcloudapi.com/v1",
                api_key_env="ZCLOUD_API_KEY",
                max_tokens=100,
            ),
        )

    # 命令行参数覆盖
    if args.max_workers is not None:
        config.max_workers = args.max_workers
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval

    # 数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        root = Path(__file__).resolve().parent
        # 尝试多个可能的位置
        for candidate in [
            root / ".hf313/hub/datasets--skylenage-ai--HLE-Verified/snapshots/0bc83643672d4f68a5f89998617a639d85e7318b/data",
            root / ".hf-eval/hub/datasets--skylenage-ai--HLE-Verified/snapshots/0bc83643672d4f68a5f89998617a639d85e7318b/data",
        ]:
            if candidate.is_dir():
                data_dir = candidate
                break
        else:
            console.print("[red]Error: HLE data directory not found[/red]")
            return 1

    # 运行评估
    evaluator = HLELiteEvaluator(
        config,
        data_dir,
        args.output_dir,
        checkpoint_file=args.checkpoint_file,
        results_file=args.results_file,
        metrics_file=args.metrics_file,
    )
    evaluator.run()

    return 0

if __name__ == "__main__":
    sys.exit(main())
