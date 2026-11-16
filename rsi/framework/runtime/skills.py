"""Explicit skill references and fingerprints for the active framework."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.contracts import METHOD_IDS


@dataclass(frozen=True)
class SkillRef:
    method_id: str
    ref: str
    kind: str
    fingerprint: str
    path: Path

    def to_manifest(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "ref": self.ref,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
        }


class SkillRegistry:
    def __init__(
        self,
        skills_root: str | Path,
        *,
        allowed_method_ids: frozenset[str] | None = None,
    ) -> None:
        self.skills_root = Path(skills_root).resolve()
        self.allowed_method_ids = allowed_method_ids

    def load(self, method_id: str, ref: str, *, kind: str = "method") -> SkillRef:
        if method_id not in METHOD_IDS:
            raise ValueError(f"unknown method_id {method_id!r}")
        if self.allowed_method_ids is not None and method_id not in self.allowed_method_ids:
            raise ValueError(f"skill loading is disabled for method_id {method_id!r}")
        if not ref or Path(ref).is_absolute():
            raise ValueError("skill ref must be a nonempty relative path")
        root = self.skills_root
        path = (root / ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"skill {ref!r} is unavailable within framework for {method_id}")
        digest = hashlib.sha256()
        files = sorted(path.parent.rglob("*")) if path.name == "SKILL.md" else [path]
        for item in files:
            if item.is_file() and not any(part.startswith(".") for part in item.relative_to(path.parent).parts):
                digest.update(item.relative_to(path.parent).as_posix().encode("utf-8"))
                digest.update(item.read_bytes())
        return SkillRef(method_id, ref.replace("\\", "/"), kind, digest.hexdigest(), path)

    def bundle(
        self,
        method_id: str,
        method_refs: Iterable[str],
        task_refs: Iterable[str] = (),
    ) -> tuple[SkillRef, ...]:
        refs = [
            *(self.load(method_id, ref, kind="method") for ref in method_refs),
            *(self.load(method_id, ref, kind="task") for ref in task_refs),
        ]
        if len({item.ref for item in refs}) != len(refs):
            raise ValueError("duplicate skill ref")
        return tuple(refs)
