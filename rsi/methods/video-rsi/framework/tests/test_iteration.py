import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video_rsi.iteration import CodexCLI, IterationConfig, RSIIterationController


class IterationTests(unittest.TestCase):
    def test_acceptance_uses_single_frontier_novel_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RSIIterationController(IterationConfig(Path(tmp)))
            record = controller.run_round(
                round_id="round_02",
                baseline_summary={"accepted_frontier_novel_count": 10, "cost": 1},
                candidate_summary={"accepted_frontier_novel_count": 11, "cost": 100},
            )
        self.assertEqual("accept", record["decision"])
        self.assertEqual(10, record["baseline_proxy"])
        self.assertEqual(11, record["candidate_proxy"])

    @patch("video_rsi.iteration.subprocess.Popen")
    def test_codex_requests_writable_workspace(self, popen):
        process = Mock(pid=101, returncode=0)
        process.communicate.return_value = ('{"type":"turn.completed"}\n', "")
        popen.return_value = process
        with tempfile.TemporaryDirectory() as tmp:
            result = CodexCLI(IterationConfig(Path(tmp))).execute("inspect")
        self.assertEqual(0, result["returncode"])
        command = popen.call_args.args[0]
        if "--sandbox" in command:
            sandbox_index = command.index("--sandbox")
            self.assertEqual("workspace-write", command[sandbox_index + 1])
        else:
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertIn("--skip-git-repo-check", command)

    @patch("video_rsi.iteration.subprocess.Popen")
    def test_codex_sandbox_failure_falls_back(self, popen):
        first = Mock(pid=101, returncode=1)
        first.communicate.return_value = ("", "bwrap: namespace unavailable")
        second = Mock(pid=102, returncode=0)
        second.communicate.return_value = ('{"type":"turn.completed"}\n', "")
        popen.side_effect = [first, second]
        with tempfile.TemporaryDirectory() as tmp:
            result = CodexCLI(IterationConfig(Path(tmp), sandbox_mode="workspace-write")).execute("inspect")
        self.assertEqual(0, result["returncode"])
        self.assertEqual(2, popen.call_count)
        self.assertEqual("danger-full-access", result["sandbox_mode"])

    @patch("video_rsi.iteration.subprocess.Popen")
    def test_codex_internal_sandbox_failure_with_zero_exit_falls_back(self, popen):
        first = Mock(pid=101, returncode=0)
        first.communicate.return_value = (
            '{"type":"turn.completed"}\n',
            "Failed to write file /tmp/probe.txt: namespace creation failure",
        )
        second = Mock(pid=102, returncode=0)
        second.communicate.return_value = ('{"type":"turn.completed"}\n', "")
        popen.side_effect = [first, second]
        with tempfile.TemporaryDirectory() as tmp:
            result = CodexCLI(IterationConfig(Path(tmp), sandbox_mode="workspace-write")).execute("inspect")
        self.assertEqual(2, popen.call_count)
        self.assertFalse(result["sandbox_failure"])
        self.assertEqual("danger-full-access", result["sandbox_mode"])

    @patch("video_rsi.iteration.subprocess.Popen")
    def test_codex_resume_preserves_model_and_reasoning_effort(self, popen):
        process = Mock(pid=103, returncode=0)
        process.communicate.return_value = ('{"type":"turn.completed"}\n', "")
        popen.return_value = process
        with tempfile.TemporaryDirectory() as tmp:
            config = IterationConfig(
                Path(tmp), coding_model="gpt-5.6-terra", reasoning_effort="medium"
            )
            result = CodexCLI(config).resume("session-1", "review")
        self.assertEqual(0, result["returncode"])
        command = popen.call_args.args[0]
        self.assertEqual("gpt-5.6-terra", command[command.index("--model") + 1])
        self.assertIn("model_reasoning_effort=medium", command)

    @patch.object(CodexCLI, "resume")
    @patch.object(CodexCLI, "execute")
    def test_proposal_uses_stateful_multi_turn_dialogue(self, execute, resume):
        execute.return_value = {
            "returncode": 0, "stdout": "diagnosis", "stderr": "",
            "session_id": "sess-1", "events": [],
        }
        resume.side_effect = [
            {"returncode": 0, "stdout": "implemented", "stderr": "",
             "session_id": "sess-1", "events": []},
            {"returncode": 0, "stdout": "reviewed DECISION: CONTINUE", "stderr": "",
             "session_id": "sess-1", "events": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            controller = RSIIterationController(IterationConfig(Path(tmp)))
            result = controller.propose_change(
                round_id="round_01", baseline_summary={}, logs=(), examples=()
            )
        self.assertEqual(3, result["turns_completed"])
        self.assertEqual(2, resume.call_count)
        self.assertIn("implemented", result["stdout"])

    @patch.object(CodexCLI, "resume")
    @patch.object(CodexCLI, "execute")
    def test_turn_two_transport_failure_uses_short_checkpoint_recovery(self, execute, resume):
        def implementation_failure(*_args, **_kwargs):
            return {
                "returncode": 1, "stdout": "stream disconnected before completion",
                "stderr": "", "session_id": "sess-1", "events": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "pipeline.py").write_text("before = 1\n")

            def recovery(prompt):
                if "interrupted" in prompt:
                    (root / "src" / "pipeline.py").write_text("after = 2\n")
                    return {
                        "returncode": 0, "stdout": "CHECKPOINT_DONE", "stderr": "",
                        "session_id": "sess-2", "events": [],
                    }
                return {
                    "returncode": 0, "stdout": "diagnosis", "stderr": "",
                    "session_id": "sess-1", "events": [],
                }

            execute.side_effect = recovery
            resume.side_effect = lambda *_args, **_kwargs: (
                implementation_failure()
                if resume.call_count == 1 else {
                    "returncode": 0, "stdout": "reviewed DECISION: CONTINUE",
                    "stderr": "", "session_id": "sess-2", "events": [],
                }
            )
            controller = RSIIterationController(IterationConfig(root))
            result = controller.propose_change(
                round_id="round_01", baseline_summary={}, logs=(), examples=()
            )
        self.assertEqual(0, result["returncode"])
        self.assertEqual(2, resume.call_count)
        self.assertEqual(2, execute.call_count)
        self.assertIn("CHECKPOINT_DONE", result["stdout"])

    @patch.object(CodexCLI, "resume")
    @patch.object(CodexCLI, "execute")
    def test_turn_three_transport_failure_defers_to_controller_validation(self, execute, resume):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "pipeline.py").write_text("before = 1\n")
            execute.return_value = {
                "returncode": 0, "stdout": "diagnosis", "stderr": "",
                "session_id": "sess-1", "events": [],
            }

            def implement(*_args, **_kwargs):
                (root / "src" / "pipeline.py").write_text("after = 2\n")
                return {
                    "returncode": 0, "stdout": "implemented", "stderr": "",
                    "session_id": "sess-1", "events": [],
                }

            resume.side_effect = lambda *_args, **_kwargs: (
                implement()
                if resume.call_count == 1 else {
                    "returncode": 1, "stdout": "stream disconnected before completion",
                    "stderr": "", "session_id": "sess-1", "events": [],
                }
            )
            controller = RSIIterationController(IterationConfig(root))
            result = controller.propose_change(
                round_id="round_01", baseline_summary={}, logs=(), examples=()
            )
        self.assertEqual(0, result["returncode"])
        self.assertTrue(result.get("controller_recovered"))


if __name__ == "__main__":
    unittest.main()
