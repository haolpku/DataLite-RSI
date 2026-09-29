"""Task skill packs: reusable, versioned guidance for pipeline assembly.

A skill is deliberately data, not executable hidden logic.  It tells the
evolver what a task means, which evidence is required, and which operators or
checks are useful.  Pipelines may reuse a skill while evolving independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    version: str
    task_types: tuple[str, ...]
    instructions: str
    preferred_operators: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    evaluation_checks: tuple[str, ...] = ()
    forbidden_shortcuts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.skill_id}@{self.version}"

    @property
    def fingerprint(self) -> str:
        value = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "task_types": list(self.task_types),
            "instructions": self.instructions,
            "preferred_operators": list(self.preferred_operators),
            "required_capabilities": list(self.required_capabilities),
            "evaluation_checks": list(self.evaluation_checks),
            "forbidden_shortcuts": list(self.forbidden_shortcuts),
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }

    def render_context(self) -> str:
        checks = "\n".join(f"- {item}" for item in self.evaluation_checks) or "- none specified"
        forbidden = ", ".join(self.forbidden_shortcuts) or "none specified"
        return (
            f"Skill: {self.ref}\n"
            f"Task types: {', '.join(self.task_types)}\n"
            f"Instructions:\n{self.instructions}\n"
            f"Required checks:\n{checks}\n"
            f"Forbidden shortcuts: {forbidden}"
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, skill: SkillSpec, *, replace: bool = False) -> SkillSpec:
        if skill.ref in self._skills and not replace:
            raise KeyError(f"skill {skill.ref!r} is already registered")
        self._skills[skill.ref] = skill
        return skill

    def get(self, ref: str) -> SkillSpec:
        if ref in self._skills:
            return self._skills[ref]
        matches = [s for s in self._skills.values() if s.skill_id == ref]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"unknown or ambiguous skill {ref!r}; available={sorted(self._skills)}")

    def route(self, task_type: str, *, version: str | None = None) -> SkillSpec:
        matches = [s for s in self._skills.values() if task_type in s.task_types]
        if version is not None:
            matches = [s for s in matches if s.version == version]
        if not matches:
            raise KeyError(f"no skill registered for task_type={task_type!r}")
        return sorted(matches, key=lambda s: s.version)[-1]

    def list(self, task_type: str | None = None) -> list[dict[str, Any]]:
        skills: Iterable[SkillSpec] = self._skills.values()
        if task_type is not None:
            skills = [s for s in skills if task_type in s.task_types]
        return [s.to_dict() for s in sorted(skills, key=lambda x: x.ref)]

    def load(self, path: str | Path, *, replace: bool = False) -> SkillSpec:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        skill = SkillSpec(
            skill_id=str(value["skill_id"]),
            version=str(value["version"]),
            task_types=tuple(value.get("task_types", ())),
            instructions=str(value.get("instructions", "")),
            preferred_operators=tuple(value.get("preferred_operators", ())),
            required_capabilities=tuple(value.get("required_capabilities", ())),
            evaluation_checks=tuple(value.get("evaluation_checks", ())),
            forbidden_shortcuts=tuple(value.get("forbidden_shortcuts", ())),
            metadata=dict(value.get("metadata", {})),
        )
        return self.register(skill, replace=replace)


SKILL_REGISTRY = SkillRegistry()



