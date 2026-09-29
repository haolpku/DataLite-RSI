import json
import tempfile
import unittest
from pathlib import Path

from video_rsi.corpus import load_caption_result


class CorpusTests(unittest.TestCase):
    def test_load_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_key" / "result.json"
            path.parent.mkdir()
            payload = {
                "status": "completed",
                "video": {"video_id": "v", "duration_sec": 10},
                "expected_segment_count": 1,
                "segments": [
                    {
                        "segment": {
                            "index": 0,
                            "start_sec": 0,
                            "end_sec": 10,
                            "duration_sec": 10,
                        },
                        "caption": {"minute_caption": "[00:00-00:10] A man walks."},
                        "entities": {
                            "new_entities": [],
                            "updated_entities": [],
                            "events": [],
                        },
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = load_caption_result(path)
            self.assertEqual("video_key", record.video_key)
            self.assertEqual(1, len(record.state["observations"]))


if __name__ == "__main__":
    unittest.main()

