from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SceneCategory = Literal[
    "object",
    "food",
    "human",
    "animal",
    "vehicle",
    "indoor",
    "outdoor",
    "nature",
    "architecture",
    "chart",
    "diagram",
    "infographic",
    "poster",
    "ui",
]

SceneStyle = Literal[
    "photo",
    "studio",
    "illustration",
    "render3d",
    "anime",
    "sketch",
    "poster",
    "screenshot",
    "chart",
]

SceneComposition = Literal["single_subject", "few_subjects", "many_subjects"]

SceneLighting = Literal["normal", "low_light", "backlit", "haze", "motion_blur"]


class SceneDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    prompt: str
    category: SceneCategory
    style: SceneStyle
    has_text: bool = False
    has_human: bool = False
    composition: Optional[SceneComposition] = None
    lighting: SceneLighting = "normal"

    @property
    def ref(self) -> str:
        return self.scene_id


class EditDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edit_id: str
    subcategory: str
    description: str

    @property
    def ref(self) -> str:
        return self.edit_id


class CompatibilityDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compatibility_id: str
    edit_ref: str
    allowed_categories: list[str]
    allowed_scene_refs: list[str] = Field(default_factory=list)

    @property
    def ref(self) -> str:
        return self.compatibility_id


class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    description: str
    scene_registry: str
    edit_registry: str
    compatibility_registry: str
    scene_refs: list[str]
    edit_refs: list[str]
    compatibility_refs: list[str]

    @property
    def ref(self) -> str:
        return self.snapshot_id
