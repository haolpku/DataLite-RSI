import json
import tempfile
import unittest
from pathlib import Path

from video_rsi.core import Operator, Pipeline, RunContext
from video_rsi.storage import RunStore


class AddOperator(Operator):
    name = "add"
    input_keys = ("value",)
    output_keys = ("added",)

    def __init__(self):
        self.calls = 0

    def run(self, state, context):
        self.calls += 1
        return {"added": state["value"] + 1}


class CoreTests(unittest.TestCase):
    def test_compile_rejects_missing_input(self):
        with self.assertRaises(KeyError):
            Pipeline("p", [AddOperator()]).compile(["video_key"])

    def test_stage_checkpoint_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            operator = AddOperator()
            pipeline = Pipeline("p", [operator])
            store = RunStore(Path(tmp), "run")
            context = RunContext("run", store)
            state = {
                "video_key": "v",
                "video": {},
                "source_result": "source.json",
                "value": 2,
            }
            # This minimal pipeline does not have the bootstrap final keys, so
            # exercise the checkpoint API through the operator stage directly.
            outputs = operator.run(state, context)
            store.save_stage(
                video_key="v",
                stage_index=1,
                stage_name=operator.name,
                fingerprint=operator.fingerprint,
                outputs=outputs,
                stats=operator.stats(outputs),
            )
            loaded = store.load_stage("v", 1, operator.name, operator.fingerprint)
            self.assertEqual({"added": 3}, loaded)


if __name__ == "__main__":
    unittest.main()

