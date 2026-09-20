from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RubricAxisDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis_id: str
    name_zh: str
    name_en: str
    description: str
    evidence_requirements: list[str] = Field(default_factory=list)
    weight: float = 1.0
    minimum_score: float = 3.0


class RubricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    description: str
    judge_model: str
    scale: str = "1-5 Likert"
    environment_check: list[str]
    axes: list[RubricAxisDefinition]
    minimum_average: float = 3.5

    @property
    def ref(self) -> str:
        return self.rubric_id
