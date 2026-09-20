from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_YAML = False


SettingId = Literal[
    "random",
    "stratified",
    "evolving",
    "evolving-no-rewrite",
    "policy-evolving",
]

DEFAULT_COVERAGE_AXES = ["taxonomy", "category", "style", "composition"]


class BudgetConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    max_api_calls: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=50, ge=1)
    max_batches: int = Field(default=20, ge=1)
    target_goodcase: Optional[int] = Field(default=None, ge=1)


class SeedPoolConfig(BaseModel):

    model_config = ConfigDict(extra="allow")

    seeds_path: str
    base_image_count: Optional[int] = Field(default=None, ge=1)


class GeneratorConfig(BaseModel):

    model_config = ConfigDict(extra="allow")

    name: str
    kind: str = "openai_compat"
    model: Optional[str] = None
    weight: float = 1.0


class VerifierConfig(BaseModel):

    model_config = ConfigDict(extra="allow")

    name: str
    model: Optional[str] = None
    rubric_id: str = "if_vc_vq"


class GoodCaseThreshold(BaseModel):

    model_config = ConfigDict(extra="forbid")

    aggregate_score: float = 3.5
    badcase_max_score: float = 2.5


class AgentPolicyConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    rewrite_share: float = 0.3
    ablation: Optional[Literal["no_rewrite"]] = None


class EvolveConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    gate_min_support_batches: int = Field(default=2, ge=1)
    max_policies: int = Field(default=8, ge=1)
    retire_after_untested: int = Field(default=2, ge=1)


class CampaignSpec(BaseModel):

    model_config = ConfigDict(extra="allow")

    campaign_id: str
    task: Literal["image_edit", "image_t2i", "image_multiturn"] = "image_edit"
    setting_id: SettingId = "evolving"

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    environment_snapshot: Optional[str] = None
    restrict_edits: list[str] = Field(default_factory=list)
    restrict_scenes: list[str] = Field(default_factory=list)
    seed_pool: Optional[SeedPoolConfig] = None
    coverage_axes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_COVERAGE_AXES)
    )

    generator_pool: list[GeneratorConfig] = Field(default_factory=list)
    verifier_stack: list[VerifierConfig] = Field(default_factory=list)

    goodcase_threshold: GoodCaseThreshold = Field(default_factory=GoodCaseThreshold)
    agent_policy: AgentPolicyConfig = Field(default_factory=AgentPolicyConfig)
    evolve: EvolveConfig = Field(default_factory=EvolveConfig)

    random_seeds: list[int] = Field(default_factory=lambda: [42])
    notes: dict[str, Any] = Field(default_factory=dict)


    @model_validator(mode="after")
    def _check(self) -> "CampaignSpec":
        if not self.generator_pool:
            raise ValueError("CampaignSpec.generator_pool must have >=1 entry")
        if not self.verifier_stack:
            raise ValueError("CampaignSpec.verifier_stack must have >=1 entry")
        if not self.coverage_axes:
            raise ValueError("CampaignSpec.coverage_axes must have >=1 axis")
        if not self.random_seeds:
            raise ValueError("CampaignSpec.random_seeds must have >=1 seed")
        if (
            self.goodcase_threshold.badcase_max_score
            > self.goodcase_threshold.aggregate_score
        ):
            raise ValueError(
                "GoodCaseThreshold: badcase_max_score must be <= aggregate_score"
            )
        return self


    def dump_yaml(self) -> str:
        if not _HAS_YAML:
            return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)
        return yaml.safe_dump(  # type: ignore[union-attr]
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )


def load_campaign(path: str | Path) -> CampaignSpec:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CampaignSpec file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yml", ".yaml"}:
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML not installed but campaign file is yaml — "
                "either `pip install pyyaml` or convert to .json"
            )
        data = yaml.safe_load(text)  # type: ignore[union-attr]
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if _HAS_YAML:
            try:
                data = yaml.safe_load(text)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                data = json.loads(text)
        else:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"CampaignSpec must be a mapping, got {type(data).__name__}")
    return CampaignSpec.model_validate(data)
