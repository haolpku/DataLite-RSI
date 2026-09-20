from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(str, Enum):
    T2I = "t2i"
    EDIT = "edit"
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    CONTROL = "control"
    COMPOSITION = "composition"
    MULTITURN = "multiturn"
    EDIT_CHAIN = "edit_chain"


class Judgment(str, Enum):
    PENDING = "Pending"
    GOOD = "GoodCase"
    BAD = "BadCase"
    UNKNOWN = "Unknown"


class ConstraintOp(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    PRESERVE = "preserve"


class ConstraintType(str, Enum):
    OBJECT = "object"
    TEXT = "text"
    LAYOUT = "layout"
    STYLE = "style"
    LIGHTING = "lighting"
    HUMAN = "human"
    DATA = "data"
    BACKGROUND = "background"
    OTHER = "other"


class ImageSource(str, Enum):
    GENERATED = "generated"
    EXTERNAL = "external"
    WEB = "web"
    MANUAL = "manual"
    REFERENCE = "reference"
    CONTROL = "control"
    MASK = "mask"


class ControlSignalType(str, Enum):
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    SEG = "seg"
    LINEART = "lineart"
    NORMAL = "normal"
    SCRIBBLE = "scribble"
    OTHER = "other"


class ReferenceRole(str, Enum):
    SUBJECT = "subject"
    STYLE = "style"
    GENERAL = "general"
    POSE = "pose"
    FACE = "face"


class BBox(BaseModel):

    model_config = ConfigDict(extra="allow")

    x: float
    y: float
    w: float
    h: float


class ImageRef(BaseModel):

    model_config = ConfigDict(extra="allow")

    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    aspect_ratio: Optional[str] = None
    source: Optional[ImageSource] = None
    prompt: Optional[str] = None
    method: Optional[str] = None


class ControlSignal(BaseModel):

    model_config = ConfigDict(extra="allow")

    type: ControlSignalType
    image: ImageRef
    weight: Optional[float] = None


class ReferenceImage(BaseModel):

    model_config = ConfigDict(extra="allow")

    role: ReferenceRole = ReferenceRole.GENERAL
    image: ImageRef
    weight: Optional[float] = None
    description: Optional[str] = None


class Constraint(BaseModel):

    model_config = ConfigDict(extra="allow")

    op: ConstraintOp
    type: ConstraintType = ConstraintType.OTHER
    target: str
    value: Optional[str] = None
    region: Optional[BBox] = None
    note: Optional[str] = None


class Sampling(BaseModel):

    model_config = ConfigDict(extra="allow")

    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    extras: dict[str, Any] = Field(default_factory=dict)


class Inputs(BaseModel):

    model_config = ConfigDict(extra="allow")

    base_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[ImageRef] = None
    mask: Optional[ImageRef] = None
    control_signals: list[ControlSignal] = Field(default_factory=list)
    reference_images: list[ReferenceImage] = Field(default_factory=list)
    sampling: Optional[Sampling] = None


class StepInputs(BaseModel):

    model_config = ConfigDict(extra="allow")

    init_image: Optional[ImageRef] = None
    mask: Optional[ImageRef] = None
    control_signals: Optional[list[ControlSignal]] = None
    reference_images: Optional[list[ReferenceImage]] = None
    sampling: Optional[Sampling] = None


class StepCost(BaseModel):

    model_config = ConfigDict(extra="allow")

    step: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_tokens_details: Optional[dict[str, Any]] = None
    quota: Optional[float] = None
    latency_s: Optional[float] = None


class CostMeta(BaseModel):

    model_config = ConfigDict(extra="allow")

    total_tokens: Optional[int] = None
    total_quota: Optional[float] = None
    latency_total_s: Optional[float] = None
    steps: list[StepCost] = Field(default_factory=list)

    def add(self, step: StepCost) -> None:
        self.steps.append(step)
        if step.total_tokens is not None:
            self.total_tokens = (self.total_tokens or 0) + step.total_tokens
        if step.quota is not None:
            self.total_quota = (self.total_quota or 0.0) + step.quota
        if step.latency_s is not None:
            self.latency_total_s = (self.latency_total_s or 0.0) + step.latency_s


class Instruction(BaseModel):

    model_config = ConfigDict(extra="allow")

    zh: Optional[str] = None
    en: Optional[str] = None


class DeltaStep(BaseModel):

    model_config = ConfigDict(extra="allow")

    index: int = 1

    instruction: Instruction = Field(default_factory=Instruction)

    preserve_text: list[str] = Field(default_factory=list)
    change_text: list[str] = Field(default_factory=list)
    constraints: Optional[list[Constraint]] = None

    thinking: Optional[str] = None

    step_inputs: Optional[StepInputs] = None

    image: Optional[ImageRef] = None
    cost: Optional[StepCost] = None
    duration_ms: Optional[int] = None

    raw: dict[str, Any] = Field(default_factory=dict)


class AxisScore(BaseModel):

    model_config = ConfigDict(extra="allow")

    score: float
    reason: Optional[str] = None
    rubric_id: Optional[str] = None


class HumanRating(BaseModel):
    model_config = ConfigDict(extra="allow")

    rater: Optional[str] = None
    score: Optional[float] = None
    judgment: Optional[Judgment] = None
    notes: Optional[str] = None
    rated_at: Optional[datetime] = None


class VerifyReport(BaseModel):

    model_config = ConfigDict(extra="allow")

    rubric_id: str = "if_vc_vq"
    axes: dict[str, AxisScore] = Field(default_factory=dict)
    aggregate: Optional[float] = None
    judgment: Optional[Judgment] = None
    verifier_model: Optional[str] = None
    human_rating: Optional[HumanRating] = None
    failure_tags: list[str] = Field(default_factory=list)


class SampleMeta(BaseModel):

    model_config = ConfigDict(extra="allow")

    collector: Optional[str] = None
    model_before: Optional[str] = None
    model_edit: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    notes: Optional[str] = None
    judgment: Judgment = Judgment.PENDING
    tags: list[str] = Field(default_factory=list)
    source_format: Optional[str] = None
    run_id: Optional[str] = None
    pair_meta: dict[str, Any] = Field(default_factory=dict)

    slice_id: Optional[str] = None
    batch_id: Optional[str] = None
    setting_id: Optional[str] = None


class Sample(BaseModel):

    model_config = ConfigDict(extra="allow")

    sample_id: str
    schema_version: Literal["1.0"] = "1.0"
    task_type: TaskType

    parent_sample_id: Optional[str] = None
    derived_from: list[str] = Field(default_factory=list)
    rewrite_parent_id: Optional[str] = None

    category: str
    subcategory: Optional[str] = None
    edit_codes: list[str] = Field(default_factory=list)

    inputs: Inputs = Field(default_factory=Inputs)
    steps: list[DeltaStep] = Field(default_factory=list)
    final_image: Optional[ImageRef] = None

    verification: Optional[VerifyReport] = None
    cost: CostMeta = Field(default_factory=CostMeta)
    meta: SampleMeta = Field(default_factory=SampleMeta)


    @property
    def K0(self) -> Optional[ImageRef]:  # noqa: N802 — keep paper notation
        return self.inputs.init_image

    @property
    def before(self) -> Optional[ImageRef]:
        return self.inputs.init_image

    @property
    def after(self) -> Optional[ImageRef]:
        if self.final_image is not None:
            return self.final_image
        if self.steps:
            return self.steps[-1].image
        return None

    @property
    def rounds(self) -> list[DeltaStep]:
        return self.steps


    @model_validator(mode="after")
    def _check_invariants(self) -> "Sample":
        t = self.task_type
        n = len(self.steps)
        i = self.inputs

        def _require(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"[{t.value}] {msg}")

        if t in (
            TaskType.T2I,
            TaskType.EDIT,
            TaskType.INPAINT,
            TaskType.OUTPAINT,
            TaskType.CONTROL,
            TaskType.COMPOSITION,
        ):
            _require(n == 1, f"task must have exactly 1 step, got {n}")
        elif t in (TaskType.MULTITURN, TaskType.EDIT_CHAIN):
            _require(n >= 2, f"task must have >=2 steps, got {n}")

        if t is TaskType.T2I:
            _require(i.init_image is None, "t2i must not have init_image")
            _require(i.mask is None, "t2i must not have mask")
            _require(not i.control_signals, "t2i must not have control_signals")
            _require(not i.reference_images, "t2i must not have reference_images")

        elif t is TaskType.EDIT:
            _require(i.init_image is not None, "edit requires init_image")
            _require(i.mask is None, "edit must not have mask (use inpaint instead)")

        elif t in (TaskType.INPAINT, TaskType.OUTPAINT):
            _require(i.init_image is not None, f"{t.value} requires init_image")
            _require(i.mask is not None, f"{t.value} requires mask")

        elif t is TaskType.CONTROL:
            _require(i.init_image is None, "control must not have init_image")
            _require(
                len(i.control_signals) >= 1,
                "control requires >=1 control_signals",
            )

        elif t is TaskType.COMPOSITION:
            _require(
                len(i.reference_images) >= 1,
                "composition requires >=1 reference_images",
            )

        elif t is TaskType.MULTITURN:
            _require(i.init_image is None, "multiturn must not have init_image")
            _require(i.mask is None, "multiturn must not have mask")
            _require(not i.control_signals, "multiturn must not have control_signals")
            _require(not i.reference_images, "multiturn must not have reference_images")

        elif t is TaskType.EDIT_CHAIN:
            _require(i.init_image is not None, "edit_chain requires init_image")

        if self.final_image is None and self.steps and self.steps[-1].image is not None:
            self.final_image = self.steps[-1].image

        return self
