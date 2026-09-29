"""Post-v0 autonomous RSI controller using a headless Codex CLI.

This module deliberately controls *proposal, test and acceptance*. Codex may
edit TaskSpec, PromptSpec, Operator or Pipeline code, but it does not decide
whether its own change is accepted and cannot change the frozen target or
evaluation policy.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .model_policy import ModelPolicy, V0_MODEL_POLICY
from .storage import atomic_write_json, utc_now

try:
    # The experiment package is optional for framework users.  When present,
    # it supplies the fixed-policy promotion decision used by the runner.
    from evaluation import compare_evaluations
except ImportError:  # pragma: no cover - framework-only installation
    compare_evaluations = None


@dataclass(frozen=True)
class IterationConfig:
    workspace: Path
    coding_model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"
    max_rounds: int = 1
    # One proposal is a short, stateful Codex dialogue: diagnose, implement,
    # then review/tests.  The outer RSI budget still controls how many
    # proposals are attempted; this controls depth within a proposal.
    agent_turns: int = int(os.environ.get("RSI_CODEX_AGENT_TURNS", "3"))
    fixed_budget_label: str = "probe-set-v0-fixed-budget"
    model_policy: ModelPolicy = V0_MODEL_POLICY
    # Set RSI_CODEX_BIN when a site needs a private wrapper or proxy. The
    # public default remains the standard Codex executable.
    codex_bin: str = os.environ.get("RSI_CODEX_BIN", "codex")
    timeout_sec: float = float(os.environ.get("RSI_CODEX_TIMEOUT_SEC", "600"))
    max_retries: int = int(os.environ.get("RSI_CODEX_RETRIES", "2"))
    # The shared GPU servers disallow unprivileged bwrap namespaces.  The
    # candidate workspace is isolated by the outer experiment directory, so
    # start the Codex session directly in full-access mode; resume cannot
    # change a session's sandbox after it has been created.
    sandbox_mode: str = os.environ.get("RSI_CODEX_SANDBOX", "danger-full-access")
    fallback_sandbox_mode: str = os.environ.get(
        "RSI_CODEX_FALLBACK_SANDBOX", "danger-full-access"
    )


class CodexCLI:
    def __init__(self, config: IterationConfig) -> None:
        self.config = config
        # The initial execution may transparently switch from the unusable
        # workspace sandbox to the server's fallback sandbox.  Resume must use
        # the same effective mode, otherwise Turn 2/3 can fail again.
        self._effective_sandbox_mode = config.sandbox_mode

    @staticmethod
    def _sandbox_failure(text: str) -> bool:
        """Detect sandbox failures even when Codex exits with status zero.

        Codex can emit a successful ``turn.completed`` event while one or more
        tool calls failed inside bwrap.  In that case checking only the process
        return code leaves the controller believing that a proposal completed.
        """
        lowered = text.lower()
        return bool(re.search(
            r"bwrap|namespace creation|non-privileged user namespaces|"
            r"approval policy is never|failed to write file|"
            r"operation not permitted|permission denied|read-only filesystem",
            lowered,
        ))

    @staticmethod
    def transport_failure(text: str) -> bool:
        """Recognise incomplete streamed responses from a provider or proxy.

        A completed short request proves neither a long coding turn nor an
        SSE connection is healthy.  Keep this distinct from a model/tool
        failure: callers can recover a streamed turn from the checkpointed
        workspace without treating it as a bad proposed change.
        """
        return bool(re.search(
            r"stream disconnected|stream closed before response\.completed|"
            r"reconnecting\.\.\.|connection (?:reset|closed|refused)|"
            r"failed to connect|status (?:429|502|503|504)|service unavailable",
            text.lower(),
        ))

    def execute(self, prompt: str) -> dict[str, Any]:
        def run(mode: str) -> tuple[list[str], subprocess.CompletedProcess[str]]:
            command = [
                self.config.codex_bin,
                "exec",
                "--json",
                "--skip-git-repo-check",
                # The server kernel rejects bwrap even for the
                # danger-full-access policy.  This flag bypasses Codex's
                # internal sandbox entirely; the controller still confines
                # edits to the per-round candidate workspace.
                *( ["--dangerously-bypass-approvals-and-sandbox"]
                   if mode == "danger-full-access" else ["--sandbox", mode] ),
                "--cd",
                str(self.config.workspace),
                "--model",
                self.config.coding_model,
                "-c",
                f"model_reasoning_effort={self.config.reasoning_effort}",
                # Use stdin rather than a very long argv element.  This also
                # makes the headless invocation deterministic under zsh/bash.
                "-",
            ]
            process = subprocess.Popen(
                command,
                cwd=str(self.config.workspace),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    input=prompt, timeout=self.config.timeout_sec
                )
                completed = subprocess.CompletedProcess(
                    command, process.returncode, stdout=stdout, stderr=stderr
                )
            except subprocess.TimeoutExpired as exc:
                # Killing only the wrapper leaves the real Codex process and
                # its thread-store writer alive.  Terminate the complete
                # process group and reap it before retrying the logical round.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                completed = subprocess.CompletedProcess(
                    command, 124,
                    stdout=stdout or "",
                    stderr=(stderr or "") +
                    f"\nCodex process group timed out and was reaped after {self.config.timeout_sec:.0f}s",
                )
            return command, completed

        attempts: list[dict[str, Any]] = []
        command: list[str] = []
        completed: subprocess.CompletedProcess[str] | None = None
        mode = self.config.sandbox_mode
        # Retry transient transport failures before spending the iteration.
        # Each attempt is preserved in the audit record.
        for retry_index in range(self.config.max_retries + 1):
            command, completed = run(mode)
            attempts.append({"sandbox": mode, "retry": retry_index,
                             "returncode": completed.returncode,
                             "stderr": completed.stderr,
                             "stdout": completed.stdout})
            if completed.returncode == 0:
                break
            # A bwrap/namespace failure is deterministic for this host; move
            # directly to the fallback sandbox instead of burning a retry.
            if self._sandbox_failure(completed.stdout + "\n" + completed.stderr):
                break
        assert completed is not None
        sandbox_error = any(
            self._sandbox_failure(str(item["stdout"]) + "\n" + str(item["stderr"]))
            for item in attempts
        )
        # Some Codex tool failures are reported in the JSON event stream while
        # the outer CLI still returns 0.  Retry those with the configured
        # fallback sandbox just as we do for a non-zero process exit.
        if sandbox_error and self.config.fallback_sandbox_mode != mode:
            mode = self.config.fallback_sandbox_mode
            for retry_index in range(self.config.max_retries + 1):
                command, completed = run(mode)
                attempts.append({"sandbox": mode, "retry": retry_index,
                                 "returncode": completed.returncode,
                                 "stderr": completed.stderr,
                                 "stdout": completed.stdout})
                if completed.returncode == 0:
                    break
        # Only diagnostics from the *last* attempt determine whether the
        # proposal is still unusable.  The first workspace-write attempt is
        # intentionally retained in ``attempts`` for audit, but its bwrap
        # errors must not poison a successful fallback run.  In particular,
        # Codex may quote an earlier tool error in its final JSON stdout.
        final_sandbox_error = self._sandbox_failure(completed.stderr)
        session_id = None
        events = []
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            # `codex exec --json` identifies a resumable conversation with a
            # `thread.started.thread_id` event, not necessarily a top-level
            # `session_id`.  Accept both schemas (and the occasional `id`).
            session_id = (
                event.get("session_id")
                or event.get("thread_id")
                or (event.get("thread") or {}).get("id")
                or event.get("id")
                or session_id
            )
        self._effective_sandbox_mode = mode
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "events": events,
            "session_id": session_id,
            "sandbox_mode": mode,
            "attempts": attempts,
            "sandbox_failure": final_sandbox_error,
            "fallback_used": len(attempts) > 1,
        }

    def resume(self, session_id: str, prompt: str = "Continue the interrupted RSI change.") -> dict[str, Any]:
        command = [
            self.config.codex_bin, "exec", "resume",
            # A resumed CLI invocation otherwise reloads the wrapper's default
            # model from config.toml.  Keep every turn on the model selected by
            # the evolution controller; mixing Terra and Sol can both alter the
            # proposal and fail when the provider has no channel for the
            # wrapper default.
            "--model", self.config.coding_model,
            "-c", f"model_reasoning_effort={self.config.reasoning_effort}",
            session_id, "--json",
            *( ["--dangerously-bypass-approvals-and-sandbox"]
               if self._effective_sandbox_mode == "danger-full-access"
               else [] ),
            prompt,
        ]
        attempts: list[dict[str, Any]] = []
        completed: subprocess.CompletedProcess[str] | None = None
        for retry_index in range(self.config.max_retries + 1):
            process = subprocess.Popen(
                command,
                cwd=str(self.config.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_sec)
                completed = subprocess.CompletedProcess(
                    command, process.returncode, stdout=stdout, stderr=stderr
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                completed = subprocess.CompletedProcess(
                    command, 124, stdout=stdout or "",
                    stderr=(stderr or "") +
                    f"\nCodex resume process group timed out and was reaped after {self.config.timeout_sec:.0f}s",
                )
            attempts.append({
                "retry": retry_index,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
            if completed.returncode == 0:
                break
            # The complete process group has been reaped; give the persistent
            # thread store a moment to release its writer before resuming.
            time.sleep(3)
        assert completed is not None
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "session_id": session_id,
            "events": events,
            "attempts": attempts,
        }


class RSIIterationController:
    """Run one candidate-code round; evaluation remains an injected command."""

    def __init__(self, config: IterationConfig) -> None:
        self.config = config
        self.codex = CodexCLI(config)

    @staticmethod
    def _editable_fingerprint(workspace: Path) -> str:
        """Fingerprint only source surfaces Codex is allowed to evolve."""
        digest = hashlib.sha256()
        for relative in ("src", "tests", "configs", "docs"):
            root = workspace / relative
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                digest.update(str(path.relative_to(workspace)).encode("utf-8"))
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _recover_interrupted_implementation(
        self, round_id: str, failed_turn: int
    ) -> dict[str, Any]:
        """Finish a transport-interrupted code turn in a fresh, short session.

        It deliberately avoids asking the new session to redo diagnosis or
        execute the expensive test suite.  The outer controller owns tests
        and the real-video probe, so this is a small idempotent checkpoint
        transaction rather than a second autonomous iteration.
        """
        return self.codex.execute(
            "A previous Codex stream was interrupted during Turn "
            f"{failed_turn} of Video Data RSI {round_id}. Work only from the "
            "existing repository and EVOLUTION_DIAGNOSIS.md. Inspect the current "
            "diff, then complete one coherent substantive implementation if it is "
            "not already complete. Do not redo broad diagnosis and do not run the "
            "full test suite; the controller will do that after this checkpoint. "
            "Keep edits restricted to the stated pipeline/operator/task/prompt "
            "change, add or update the required structural test, and write a concise "
            "CHANGE_PROPOSAL.md ending with DECISION: CONTINUE. Reply only with "
            "CHECKPOINT_DONE when files are written."
        )

    @staticmethod
    def primary_proxy(summary: dict[str, Any]) -> int:
        """The only acceptance proxy: accepted frontier+novel count."""
        return int(summary.get("accepted_frontier_novel_count", 0))

    def build_prompt(
        self,
        *,
        round_id: str,
        baseline_summary: dict[str, Any],
        logs: Sequence[str],
        examples: Sequence[str],
    ) -> str:
        return (
            "You are the coding agent for one Video Data RSI iteration.\n"
            f"Round: {round_id}\n"
            "This is structural exploration mode when earlier rounds tied at zero "
            "accepted samples. Do not spend a round only polishing wording or "
            "tuning a lexical duplicate threshold; that check is intentionally not "
            "a hard gate. Treat recurring failures as evidence to redesign the "
            "pipeline, an Operator, a TaskSpec, or a PromptSpec. These four surfaces "
            "are peers and all are eligible for substantive improvement; choose the "
            "surface supported by the per-operator evidence.\n"
            "You may propose and implement changes at these peer levels only: "
            "TaskSpec, task-specific PromptSpec, Operator, Pipeline graph.\n"
            "Do not change the frozen target qwen3-vl-8b-instruct, the evaluation "
            "policy, or accepted historical data.\n"
            "The fixed candidate probe is local-only and uses the configured local "
            "OpenAI-compatible vLLM endpoint; do not add a dependency on the remote "
            "chat API for validation.\n"
            "Use the repository tests and inspect concrete logs/examples before editing.\n"
            "Treat the pipeline as a Lego system: a new Operator is allowed only "
            "when it is a real reusable class, decorated with OPERATOR_REGISTRY, "
            "declares input_keys/output_keys, is imported by the pipeline builder, "
            "and is connected by compile-time key contracts. A new task-specific "
            "PromptSpec must be versioned/fingerprinted and selected explicitly by "
            "task type. Adding or replacing a Lego block must include a structural "
            "test and a concise proposal entry; do not hide a new behavior in an "
            "unregistered helper.\n"
            "Use the OperatorLibrary and PipelineGraph/PipelineRegistry when a "
            "change should be reusable or task-specific. Treat v0 as immutable: "
            "extract reusable logic into a versioned library operator, then fork "
            "or assemble a task graph that references it instead of patching v0 "
            "in place. If a task has recurring failures, create a SkillSpec or "
            "update its versioned skill pack with evidence requirements, preferred "
            "operators and evaluation checks. The skill is an explicit contract "
            "for graph assembly, not hidden code.\n"
            "Perform a comprehensive diagnosis across four peer surfaces: (1) the "
            "pipeline graph and stage ordering, (2) operator inputs, outputs and "
            "failure behavior, (3) TaskSpec/task definitions and task routing, and "
            "(4) shared and task-specific prompts. Inspect actual generated-question "
            "samples, accepted samples, dropped samples, per-stage counts, error "
            "logs, model-call traces and version diffs; do not infer quality from a "
            "single aggregate number. For every major failure cluster, identify the "
            "first stage that could have prevented it and distinguish a formatting, "
            "grounding, difficulty, shortcut, diversity or orchestration failure. "
            "When the baseline has very low yield or repeated failure modes, prefer "
            "a coherent structural redesign (including a task-specific branch or a "
            "new reusable operator) over another threshold tweak. A single round may "
            "update multiple peer-level surfaces when they address one mechanism; "
            "do not make cosmetic-only edits.\n"
            f"Fixed budget: {self.config.fixed_budget_label}\n"
            f"Baseline summary: {json.dumps(baseline_summary, ensure_ascii=False)}\n"
            f"Logs:\n{chr(10).join(logs)}\n"
            f"Examples:\n{chr(10).join(examples)}\n"
            "This is turn 1 of a stateful three-turn dialogue. First inspect the "
            "repository, logs and examples and write EVOLUTION_DIAGNOSIS.md with "
            "a per-operator failure analysis and a concrete choice among pipeline, "
            "operator, task, and prompt changes. Do not make code edits yet. End "
            "with DECISION: STOP or DECISION: CONTINUE. The controller will ask "
            "you in the next turn to implement or stop."
        )

    def run_round(
        self,
        *,
        round_id: str,
        baseline_summary: dict[str, Any],
        candidate_summary: dict[str, Any],
        logs: Sequence[str] = (),
        examples: Sequence[str] = (),
    ) -> dict[str, Any]:
        baseline_proxy = self.primary_proxy(baseline_summary)
        candidate_proxy = self.primary_proxy(candidate_summary)
        if compare_evaluations and "hard_gates" in candidate_summary:
            comparison = compare_evaluations(baseline_summary, candidate_summary)
            decision = comparison["decision"]
            reason = comparison["reason"]
        else:
            decision = "accept" if candidate_proxy > baseline_proxy else "reject"
            reason = "primary_metric_increased" if decision == "accept" else "primary_metric_not_strictly_increased"
            comparison = {}
        return {
            "round_id": round_id,
            "created_at": utc_now(),
            "baseline_proxy": baseline_proxy,
            "candidate_proxy": candidate_proxy,
            "decision": decision,
            "reason": reason,
            "fixed_budget": self.config.fixed_budget_label,
            "model_policy_version": self.config.model_policy.version,
            "model_policy_fingerprint": self.config.model_policy.fingerprint,
            "coding_model": self.config.coding_model,
            "reasoning_effort": self.config.reasoning_effort,
            "comparison": comparison,
        }

    def propose_change(
        self,
        *,
        round_id: str,
        baseline_summary: dict[str, Any],
        logs: Sequence[str] = (),
        examples: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Run a stateful diagnose→implement→review Codex dialogue.

        The first turn produces a diagnosis without editing.  Follow-up turns
        resume the same Codex session so the agent can challenge its own plan,
        implement structural changes, run tests and repair regressions before
        the controller evaluates the candidate workspace.
        """
        prompt = self.build_prompt(
            round_id=round_id,
            baseline_summary=baseline_summary,
            logs=logs,
            examples=examples,
        )
        first = self.codex.execute(prompt)
        dialogue = [{"turn": 1, **first}]
        latest = first
        session_id = first.get("session_id")
        initial_fingerprint = self._editable_fingerprint(self.config.workspace)
        if self.config.agent_turns < 1:
            return first
        followups = (
            "Turn 2: Re-read EVOLUTION_DIAGNOSIS.md and challenge your proposal. "
            "If CONTINUE is justified, implement a coherent, substantive change "
            "across whichever pipeline/operator/task/prompt surfaces are needed. "
            "You may fork a task-specific graph or add a reusable Lego operator; "
            "do not limit yourself to wording edits or a single threshold. Add "
            "structural tests and update CHANGE_PROPOSAL.md. If STOP is justified, "
            "leave code untouched and record DECISION: STOP.",
            "Turn 3: Act as your own reviewer. Inspect the complete diff and run "
            "the full test suite, including graph compilation, operator input/output "
            "contracts, task-route coverage and the one-record smoke path. Fix any "
            "regressions, missing imports, schema wiring or resume/checkpoint issues. "
            "The controller will still run the fixed real-video probe before promotion, "
            "so make the candidate executable rather than merely syntactically valid. "
            "Confirm the final change is substantive and reusable, then finalize "
            "CHANGE_PROPOSAL.md with DECISION: CONTINUE or DECISION: STOP and a "
            "short expected-effect statement.",
        )
        for turn_index, followup in enumerate(followups, start=2):
            if turn_index > self.config.agent_turns or not session_id:
                break
            try:
                latest = self.codex.resume(session_id, followup)
            except subprocess.TimeoutExpired as exc:
                latest = {
                    "command": [], "returncode": 124, "stdout": "",
                    "stderr": f"Codex follow-up timed out: {exc}",
                    "session_id": session_id, "events": [],
                }
            dialogue.append({"turn": turn_index, **latest})
            session_id = latest.get("session_id") or session_id
            if latest.get("returncode") != 0:
                combined = str(latest.get("stdout", "")) + "\n" + str(latest.get("stderr", ""))
                # Turn 2 is the only turn that must create edits.  Recover it
                # in a fresh small session rather than restarting the full
                # diagnose -> implement dialogue over an unstable SSE stream.
                if turn_index == 2 and self.codex.transport_failure(combined):
                    recovered = self._recover_interrupted_implementation(round_id, turn_index)
                    dialogue.append({"turn": "2-recovery", **recovered})
                    latest = recovered
                    session_id = recovered.get("session_id") or session_id
                    if recovered.get("returncode") == 0:
                        continue
                # Turn 3 is only self-review.  Once a successful Turn 2 has
                # checkpointed real code, the controller's own compile/tests/
                # probe are stricter and can safely replace a lost prose review.
                if (
                    turn_index == 3
                    and self.codex.transport_failure(combined)
                    and self._editable_fingerprint(self.config.workspace) != initial_fingerprint
                ):
                    latest = {
                        **latest,
                        "returncode": 0,
                        "stdout": str(latest.get("stdout", "")) +
                        "\nREVIEW_STREAM_RECOVERED: controller validation required.",
                        "stderr": str(latest.get("stderr", "")),
                        "controller_recovered": True,
                    }
                    dialogue[-1] = {"turn": turn_index, **latest}
                    break
                break
        merged = dict(latest)
        merged["session_id"] = session_id
        merged["dialogue"] = dialogue
        merged["turns_completed"] = len(dialogue)
        merged["stdout"] = "\n\n".join(str(item.get("stdout", "")) for item in dialogue)
        merged["stderr"] = "\n\n".join(str(item.get("stderr", "")) for item in dialogue if item.get("stderr"))
        merged["events"] = [event for item in dialogue for event in item.get("events", [])]
        return merged

    def save_round_record(self, root: Path, record: dict[str, Any]) -> Path:
        path = root / "iteration_rounds" / f"{record['round_id']}.json"
        atomic_write_json(path, record)
        return path
