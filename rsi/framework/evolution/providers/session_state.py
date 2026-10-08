"""Per-iteration agent session isolation and token-usage accounting."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rsi.framework.evolution.utils.config import Config
from rsi.framework.evolution.utils.logging import get_logger
from rsi.framework.evolution.telemetry.pipeline_usage import write_combined_iteration_usage

logger = get_logger("llm.session_state")


def record_codex_turn_usage(
    root: Path,
    *,
    phase: str,
    session_id: str | None,
    usage: dict[str, Any],
) -> None:
    """Persist one Codex turn's non-cumulative usage next to native session state."""
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "session_id": session_id,
        "usage": usage,
    }
    with (root / "turn_usage.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _claude_usage(root: Path) -> dict[str, Any]:
    source_files = sorted((root / "projects").glob("**/*.jsonl"))
    totals = {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
        "total_including_cache_read": 0,
        "billable_tokens_estimate_no_cache_read": 0,
    }
    models: dict[str, int] = {}
    seen_messages: set[str] = set()
    sessions: set[str] = set()
    usage_records = 0

    for source_file in source_files:
        sessions.add(source_file.stem)
        for obj in _read_jsonl(source_file):
            if obj.get("type") != "assistant" or not obj.get("timestamp"):
                continue
            message = obj.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
                continue
            usage = message["usage"]
            values = {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "cache_creation_input_tokens": int(
                    usage.get("cache_creation_input_tokens") or 0
                ),
                "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            }
            if not any(values.values()):
                continue
            dedup_key = str(
                message.get("id")
                or obj.get("requestId")
                or f"{source_file}:{obj.get('uuid')}"
            )
            if dedup_key in seen_messages:
                continue
            seen_messages.add(dedup_key)
            usage_records += 1
            for key, value in values.items():
                totals[key] += value
            totals["total_including_cache_read"] += sum(values.values())
            totals["billable_tokens_estimate_no_cache_read"] += (
                values["input_tokens"]
                + values["cache_creation_input_tokens"]
                + values["output_tokens"]
            )
            model = str(message.get("model") or "").strip()
            if model:
                models[model] = models.get(model, 0) + 1

    return {
        "backend": "claude",
        "session_state_dir": str(root),
        "source_files": [str(path) for path in source_files],
        "session_ids": sorted(sessions),
        "usage_records": usage_records,
        "models": models,
        "tokens": totals,
    }


def _opencode_usage(root: Path) -> dict[str, Any]:
    db_path = root / "data/opencode/opencode.db"
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_including_cache": 0,
        "billable_tokens_estimate_no_cache_read": 0,
    }
    sessions: list[dict[str, Any]] = []
    cost = 0.0
    status = "missing_db"

    if db_path.exists():
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(session)")
            }
            required = {
                "id",
                "title",
                "model",
                "cost",
                "tokens_input",
                "tokens_output",
                "tokens_reasoning",
                "tokens_cache_read",
                "tokens_cache_write",
            }
            if required.issubset(columns):
                rows = connection.execute(
                    """
                    SELECT id, title, model, cost, tokens_input, tokens_output,
                           tokens_reasoning, tokens_cache_read, tokens_cache_write
                    FROM session
                    ORDER BY time_created
                    """
                ).fetchall()
                status = "ok"
            else:
                rows = []
                status = "unsupported_schema"
        finally:
            connection.close()

        for row in rows:
            session = dict(row)
            sessions.append(session)
            cost += float(row["cost"] or 0)
            totals["input_tokens"] += int(row["tokens_input"] or 0)
            totals["output_tokens"] += int(row["tokens_output"] or 0)
            totals["reasoning_tokens"] += int(row["tokens_reasoning"] or 0)
            totals["cache_read_tokens"] += int(row["tokens_cache_read"] or 0)
            totals["cache_write_tokens"] += int(row["tokens_cache_write"] or 0)

    totals["total_including_cache"] = sum(
        totals[key]
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        )
    )
    totals["billable_tokens_estimate_no_cache_read"] = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["reasoning_tokens"]
        + totals["cache_write_tokens"]
    )
    return {
        "backend": "opencode",
        "session_state_dir": str(root),
        "source_db": str(db_path),
        "status": status,
        "sessions": sessions,
        "tokens": totals,
        "cost": cost,
    }


def _codex_usage(root: Path) -> dict[str, Any]:
    usage_path = root / "turn_usage.jsonl"
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "billable_tokens_estimate": 0,
    }
    sessions: set[str] = set()
    usage_records = 0
    if usage_path.is_file():
        for row in _read_jsonl(usage_path):
            usage = row.get("usage")
            if not isinstance(usage, dict):
                continue
            values = {
                key: int(usage.get(key) or 0)
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                )
            }
            if not any(values.values()):
                continue
            usage_records += 1
            for key, value in values.items():
                totals[key] += value
            # Reasoning tokens are a subset of output-token billing.
            totals["billable_tokens_estimate"] += (
                values["input_tokens"] + values["output_tokens"]
            )
            if row.get("session_id"):
                sessions.add(str(row["session_id"]))
    native_sessions = sorted((root / "sessions").glob("**/*.jsonl"))
    return {
        "backend": "codex",
        "session_state_dir": str(root),
        "source_usage_file": str(usage_path),
        "native_session_files": [str(path) for path in native_sessions],
        "session_ids": sorted(sessions),
        "usage_records": usage_records,
        "tokens": totals,
    }


@dataclass(frozen=True)
class AgentSessionStore:
    backend: str
    root: Path
    artifact_dir: Path

    def environment(self) -> dict[str, str]:
        """Return an isolated, backend-specific CLI state layout."""
        paths = [self.root, self.root / "home", self.root / "config", self.root / "cache"]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        env = {
            "HOME": str(self.root / "home"),
            "XDG_CONFIG_HOME": str(self.root / "config"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
        }
        if self.backend == "claude":
            env["CLAUDE_CONFIG_DIR"] = str(self.root)
        elif self.backend == "opencode":
            data_dir = self.root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            env["XDG_DATA_HOME"] = str(data_dir)
        elif self.backend == "codex":
            env["CODEX_HOME"] = str(self.root)
        else:
            raise ValueError(f"unsupported agent backend: {self.backend}")
        return env

    def finalize(self, *, phase: str, result: Any) -> None:
        """Record the session mapping and refresh cumulative per-iteration usage."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.artifact_dir / "agent_sessions.jsonl"
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend,
            "phase": phase,
            "session_id": getattr(result, "session_id", None),
            "success": bool(getattr(result, "success", False)),
            "error_type": getattr(result, "error_type", None),
            "session_state_dir": str(self.root),
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.backend == "claude":
            usage = _claude_usage(self.root)
        elif self.backend == "opencode":
            usage = _opencode_usage(self.root)
        elif self.backend == "codex":
            usage = _codex_usage(self.root)
        else:
            raise ValueError(f"unsupported agent backend: {self.backend}")
        usage["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.artifact_dir / "agent_token_usage.json", usage)
        write_combined_iteration_usage(self.artifact_dir)


def prepare_agent_session_store(
    agent_cfg: Config,
    *,
    backend: str,
    cwd: str,
    tool_log_dir: str | None,
) -> AgentSessionStore | None:
    """Store native CLI state under ``<iteration>/sessions/<backend>``."""
    del agent_cfg, cwd
    iteration_dir = Path(tool_log_dir).resolve() if tool_log_dir else None
    if iteration_dir is None:
        return None
    return AgentSessionStore(
        backend=backend,
        root=(iteration_dir / "sessions" / backend).resolve(),
        artifact_dir=iteration_dir,
    )


def finalize_agent_session(
    store: AgentSessionStore | None,
    *,
    phase: str,
    result: Any,
) -> Any:
    """Best-effort accounting that must never change the agent run outcome."""
    if store is not None:
        try:
            store.finalize(phase=phase, result=result)
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("记录 Agent session/token 失败（不影响主流程）：%s", exc)
    return result
