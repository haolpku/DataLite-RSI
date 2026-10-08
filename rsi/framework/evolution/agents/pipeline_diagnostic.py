from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.framework.evolution.models import PipelineConfig, ReviewResult, TaskSpec
from rsi.framework.evolution.providers.agent_runtime import AgentRuntime
from rsi.framework.evolution.providers.agent_sdk import AgentRunResult
from rsi.framework.evolution.prompts import build_pipeline_diagnostic_prompt
from rsi.framework.evolution.utils.parsing import extract_json
from rsi.framework.evolution.utils.config import Config


class PipelineDiagnosticAgent:
    """Read-only, non-scoring diagnosis of a completed pipeline execution."""

    def __init__(self, agent_cfg: Config, runtime: AgentRuntime) -> None:
        self.agent_cfg = agent_cfg
        self.runtime = runtime

    def diagnose(
        self, *, iteration_dir: Path, task: TaskSpec, observation_path: str | None,
        config: PipelineConfig, review: ReviewResult, output_path: Path,
        max_prompt_chars: int,
    ) -> dict[str, Any]:
        source_files = self._source_files(iteration_dir)
        prompt = build_pipeline_diagnostic_prompt(
            iteration_dir=str(iteration_dir),
            task=task,
            observation_path=observation_path,
            source_files=[str(path) for path in source_files],
            config=config,
            review=review,
            max_prompt_chars=max_prompt_chars,
        )
        try:
            result = self.runtime.run(
                prompt, cwd=str(iteration_dir), agent_cfg=self.agent_cfg,
                tool_log_dir=str(iteration_dir / "sessions" / "pipeline_diagnostic"),
                phase="pipeline_diagnostic",
            )
        except Exception as exc:
            result = None
            payload = {"status": "failed", "error": f"diagnostic runtime exception: {exc}"}
        if result is not None:
            payload = (
                {"status": "failed", "error": result.error or "diagnostic runtime failed"}
                if not result.success else self._parse(result)
            )
        try:
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            payload = {"status": "failed", "error": f"diagnostic artifact write failed: {exc}"}
        return payload

    @staticmethod
    def _source_files(iteration_dir: Path) -> list[Path]:
        files = [iteration_dir / "pipeline.py", iteration_dir / "decision.json"]
        operators_dir = iteration_dir / "operators"
        if operators_dir.is_dir():
            files.extend(sorted(operators_dir.glob("*.py")))
        return files

    @staticmethod
    def _parse(result: AgentRunResult) -> dict[str, Any]:
        text = result.result_text or (result.assistant_texts[-1] if result.assistant_texts else "")
        value = extract_json(text)
        if not isinstance(value, dict):
            return {"status": "failed", "error": "diagnostic agent returned no JSON object"}
        value["status"] = "ok"
        return value
