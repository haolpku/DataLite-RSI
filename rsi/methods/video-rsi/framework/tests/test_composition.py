import tempfile
import unittest
from pathlib import Path

from video_rsi.core import Operator
from video_rsi.migration import FailureSignal, PipelineMigrationPlanner
from video_rsi.operator_library import OperatorLibrary
from video_rsi.pipeline_graph import PipelineGraph, PipelineNode, PipelineRegistry
from video_rsi.registry import OperatorRegistry
from video_rsi.skills import SkillRegistry, SkillSpec


class EmitEvidence(Operator):
    name = "EmitEvidence"
    version = "1"
    input_keys = ("source",)
    output_keys = ("evidence",)

    def run(self, state, context):
        return {"evidence": state["source"]}


class EmitPool(Operator):
    name = "EmitPool"
    version = "1"
    input_keys = ("evidence",)
    output_keys = ("pool", "feedback")

    def run(self, state, context):
        return {"pool": state["evidence"], "feedback": []}


class CompositionTests(unittest.TestCase):
    def setUp(self):
        registry = OperatorRegistry()
        registry.register("EmitEvidence")(EmitEvidence)
        registry.register("EmitPool")(EmitPool)
        self.library = OperatorLibrary(registry)

    def test_graph_validates_and_instantiates(self):
        graph = PipelineGraph(
            "demo", "v1", "state_change",
            (PipelineNode("EmitEvidence@1"), PipelineNode("EmitPool@1")),
            result_keys=("pool", "feedback"),
        )
        graph.validate({"source"}, self.library)
        pipeline = graph.instantiate(library=self.library)
        self.assertEqual(pipeline.fingerprint, pipeline.fingerprint)

    def test_pipeline_registry_routes_and_roundtrips(self):
        graph = PipelineGraph("demo", "v1", "state_change", (PipelineNode("EmitEvidence@1"),))
        registry = PipelineRegistry()
        registry.register(graph)
        self.assertEqual(registry.route("state_change"), graph)
        with tempfile.TemporaryDirectory() as tmp:
            path = graph.save(Path(tmp) / "graph.json")
            loaded = PipelineRegistry().load(path)
            self.assertEqual(loaded.fingerprint, graph.fingerprint)

    def test_skill_and_migration_plan(self):
        skills = SkillRegistry()
        skill = skills.register(SkillSpec(
            "state_change", "v1", ("state_change",), "Compare the same entity before and after.",
            evaluation_checks=("same entity",), forbidden_shortcuts=("single-caption answer",),
        ))
        self.assertIn("same entity", skill.render_context())
        graph = PipelineGraph("generic", "v0", "generic", (PipelineNode("EmitEvidence@1"),))
        plan = PipelineMigrationPlanner().plan(graph, [
            FailureSignal("state_change", "too_easy", count=3, operator="QuestionGeneration")
        ])
        self.assertTrue(plan.structural_required)
        self.assertTrue(any(action.action == "fork_pipeline" for action in plan.actions))


if __name__ == "__main__":
    unittest.main()

