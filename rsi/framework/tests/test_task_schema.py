"""The published task template and schema describe the active DataFlow entry."""

from __future__ import annotations

import json
from pathlib import Path

from rsi.framework.core.contracts import METHOD_IDS, TaskEnvelope


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]


def test_schema_and_task_templates_use_only_the_active_method():
    schema = json.loads(
        (FRAMEWORK_ROOT / "schemas" / "task-envelope.schema.json").read_text(encoding="utf-8")
    )
    assert METHOD_IDS == {"dataflow-evolver"}
    assert schema["properties"]["method_id"]["const"] == "dataflow-evolver"
    for path in (FRAMEWORK_ROOT / "tasks").glob("*.json"):
        template = json.loads(path.read_text(encoding="utf-8"))
        assert template["method_id"] == "dataflow-evolver"
        assert TaskEnvelope.from_mapping(template).method_id == "dataflow-evolver"


def test_video_rsi_router_template_declares_video_contract_and_skill_profile():
    template = json.loads(
        (FRAMEWORK_ROOT / "tasks" / "video-rsi-router.json").read_text(encoding="utf-8")
    )
    assert template["input_contract"]["artifact_fields"]["video"] == "video_reference"
    assert template["metadata"]["pipeline_skill_profile"] == "video-rsi-router"
    assert template["metadata"]["candidate_evaluator"] == "video-rsi"
