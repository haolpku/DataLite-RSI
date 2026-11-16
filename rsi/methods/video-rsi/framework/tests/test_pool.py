import json
import tempfile
import unittest
from pathlib import Path

from video_rsi.pool import HighQualityPoolStore


class PoolTests(unittest.TestCase):
    def test_append_preserves_pipeline_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HighQualityPoolStore(Path(tmp))
            count = store.append(
                [
                    {
                        "sample_id": "qa_1",
                        "question": "What happens?",
                        "producer": {
                            "pipeline": "task_centric_rsi_round",
                            "pipeline_version": "0.1.0",
                            "pipeline_fingerprint": "abc",
                            "operator_provenance": [{"operator": "question_generation"}],
                        },
                    }
                ],
                rsi_round_id="round_01",
                evaluation_policy_version="eval_01",
            )
            self.assertEqual(1, count)
            record = json.loads((Path(tmp) / "pool_ledger.jsonl").read_text())
            self.assertEqual("0.1.0", record["pipeline_version"])
            self.assertEqual("abc", record["pipeline_fingerprint"])
            self.assertEqual("round_01", record["rsi_round_id"])


if __name__ == "__main__":
    unittest.main()
