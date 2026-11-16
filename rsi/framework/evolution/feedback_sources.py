"""Resolve named feedback sources without combining their scores.

The legacy downstream_eval block is a shortcut for one command source.
The per-candidate reviewer remains the selection anchor unless a custom
selection policy explicitly uses another source's results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


@dataclass(frozen=True)
class FeedbackSourceConfig:
    name: str
    kind: str  # criteria | command
    settings: dict[str, Any]
    shortcut: bool = False


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_feedback_sources(
    raw_sources: Mapping[str, Any] | None,
    downstream_shortcut: Mapping[str, Any] | None,
) -> tuple[FeedbackSourceConfig, ...]:
    """Validate names and translate the old downstream block to one source."""
    if raw_sources is None:
        raw_sources = {"criteria": {"kind": "criteria"}}
    if not isinstance(raw_sources, Mapping):
        raise ValueError("feedback_sources must be a mapping of names to settings")
    resolved: list[FeedbackSourceConfig] = []
    for name, raw in raw_sources.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid feedback source name: {name!r}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"feedback_sources.{name} must be a mapping")
        settings = dict(raw)
        if not _enabled(settings.pop("enabled", True)):
            continue
        kind = str(settings.pop("kind", "")).lower()
        if kind not in {"criteria", "command"}:
            raise ValueError(f"feedback_sources.{name}.kind must be criteria or command")
        if kind == "criteria":
            if settings:
                raise ValueError(f"feedback_sources.{name} has unknown criteria settings: {sorted(settings)}")
        else:
            allowed = {
                "command", "benchmarks", "interval", "trains_model", "baseline",
                "attribution", "timeout_sec", "bad_cases_per_benchmark", "resources",
            }
            unknown = set(settings) - allowed
            if unknown:
                raise ValueError(f"feedback_sources.{name} has unknown settings: {sorted(unknown)}")
            if not isinstance(settings.get("command"), list) or not settings["command"]:
                raise ValueError(f"feedback_sources.{name}.command must be a nonempty argv list")
            if not isinstance(settings.get("benchmarks"), list) or not settings["benchmarks"]:
                raise ValueError(f"feedback_sources.{name}.benchmarks must be a nonempty list")
            if int(settings.get("interval", 1)) < 1:
                raise ValueError(f"feedback_sources.{name}.interval must be at least 1")
        resolved.append(FeedbackSourceConfig(name, kind, settings))

    shortcut = dict(downstream_shortcut or {})
    if _enabled(shortcut.get("enabled", False)):
        name = str(shortcut.get("source_name", "downstream"))
        if not _NAME.fullmatch(name) or any(source.name == name for source in resolved):
            raise ValueError(f"downstream_eval source name is invalid or duplicated: {name!r}")
        settings = {
            key: value for key, value in shortcut.items()
            if key in {
                "command", "benchmarks", "interval", "baseline", "attribution",
                "timeout_sec", "bad_cases_per_benchmark", "resources",
            }
        }
        settings["trains_model"] = True
        resolved.append(FeedbackSourceConfig(name, "command", settings, shortcut=True))

    if sum(source.kind == "criteria" for source in resolved) != 1:
        raise ValueError("exactly one enabled criteria feedback source is required")
    return tuple(resolved)
