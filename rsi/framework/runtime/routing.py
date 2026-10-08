"""Explicit task configuration loading; no content-based method inference."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from ..core.contracts import TaskEnvelope


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in os.environ:
                raise ValueError(f"missing environment variable {key} in task config")
            return os.environ[key]
        return _ENV_REF.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_task_config(source: TaskEnvelope | Mapping[str, Any] | str | Path) -> TaskEnvelope:
    if isinstance(source, TaskEnvelope):
        return source
    if isinstance(source, Mapping):
        value = dict(source)
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            value = json.loads(text)
        elif path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML task configs require the optional PyYAML dependency") from exc
            value = yaml.safe_load(text)
        else:
            raise ValueError(f"unsupported task config extension {path.suffix!r}")
    if not isinstance(value, dict):
        raise ValueError("task config must be an object")
    return TaskEnvelope.from_mapping(_expand(value))
