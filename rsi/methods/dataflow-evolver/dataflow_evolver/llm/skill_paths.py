"""Backend-native project skill locations for PipelineAgent runtimes."""
from __future__ import annotations


SKILL_ROOTS: dict[str, str] = {
    "claude": ".claude/skills/dataflow-evolver-pipeline",
    "codex": ".agents/skills/dataflow-evolver-pipeline",
    "opencode": ".opencode/skills/dataflow-evolver-pipeline",
}


def skill_root_for_backend(backend: str) -> str:
    """Return the native project-local DataFlow skill root for ``backend``."""
    normalized = str(backend or "claude").strip().lower()
    try:
        return SKILL_ROOTS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(SKILL_ROOTS))
        raise ValueError(
            f"unknown agent backend: {normalized!r}; expected one of {supported}"
        ) from exc
