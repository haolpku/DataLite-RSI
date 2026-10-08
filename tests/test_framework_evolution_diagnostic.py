import json

import pytest

from rsi.framework.evolution.agents.pipeline_diagnostic import PipelineDiagnosticAgent
from rsi.framework.evolution.models import PipelineConfig, ReviewResult, TaskSpec
from rsi.framework.evolution.providers.agent_sdk import AgentRunResult
from rsi.framework.evolution.providers.agent_runtime import build_pipeline_diagnostic_runtime
from rsi.framework.evolution.prompts import build_pipeline_diagnostic_prompt
from rsi.framework.evolution.utils.config import Config


def test_missing_claude_sdk_has_actionable_error_without_eager_import(monkeypatch, tmp_path):
    from rsi.framework.evolution.providers import agent_sdk

    def unavailable(*args, **kwargs):
        raise ImportError("claude_agent_sdk")

    monkeypatch.setattr(agent_sdk, "_build_options", unavailable)
    with pytest.raises(RuntimeError, match="optional claude-agent-sdk package"):
        agent_sdk.run_pipeline_agent(
            "build", str(tmp_path), Config({"max_retries": 1})
        )


class Runtime:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.result


def _review():
    return ReviewResult(schema_score=0.8, relevance_score=0.8, passed=True)


def test_diagnostic_uses_independent_runtime_and_persists_json(tmp_path):
    (tmp_path / "pipeline.py").write_text("# pipeline", encoding="utf-8")
    (tmp_path / "decision.json").write_text("{}", encoding="utf-8")
    runtime = Runtime(AgentRunResult(True, result_text=json.dumps({
        "summary": "operator retained too few records",
        "findings": [],
        "contract_mismatches": [],
    })))
    cfg = Config({"backend": "codex", "read_only": True})
    agent = PipelineDiagnosticAgent(cfg, runtime)
    output = tmp_path / "pipeline_diagnostic.json"
    payload = agent.diagnose(
        iteration_dir=tmp_path,
        task=TaskSpec("task", {}, []),
        observation_path=str(tmp_path / "execution_observation.json"),
        config=PipelineConfig([{"name": "filter"}], "", "", ""),
        review=_review(),
        output_path=output,
        max_prompt_chars=4000,
    )
    assert payload["status"] == "ok"
    assert json.loads(output.read_text())["summary"] == payload["summary"]
    prompt, kwargs = runtime.calls[0]
    assert "不要给数据质量打分" in prompt
    assert kwargs["agent_cfg"] is cfg
    assert kwargs["phase"] == "pipeline_diagnostic"


def test_diagnostic_failure_is_data_not_an_exception(tmp_path):
    runtime = Runtime(AgentRunResult(False, error="unavailable"))
    agent = PipelineDiagnosticAgent(Config({}), runtime)
    payload = agent.diagnose(
        iteration_dir=tmp_path,
        task=TaskSpec("task", {}, []),
        observation_path=None,
        config=PipelineConfig([], "", "", ""),
        review=_review(),
        output_path=tmp_path / "pipeline_diagnostic.json",
        max_prompt_chars=1000,
    )
    assert payload == {"status": "failed", "error": "unavailable"}


def test_diagnostic_prompt_builder_exposes_only_diagnostic_review_context():
    review = _review()
    review.review_score = 0.77
    prompt = build_pipeline_diagnostic_prompt(
        iteration_dir="/tmp/iteration_001",
        task=TaskSpec("inspect the task", {}, []),
        observation_path="/tmp/iteration_001/execution_observation.json",
        source_files=["/tmp/iteration_001/pipeline.py"],
        config=PipelineConfig([{"name": "filter"}], "", "", ""),
        review=review,
        max_prompt_chars=4000,
    )
    assert "pipeline.py" in prompt
    assert "correctness_score" in prompt
    assert "0.77" not in prompt
    assert "不要给数据质量打分" in prompt
    assert "低基数字段分布" in prompt
