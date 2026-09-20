from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.schema import StepCost


@dataclass(slots=True)
class ServingResult:

    content: str
    cost: StepCost
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageGenResult:

    image_bytes: bytes
    mime: str = "image/jpeg"
    width: Optional[int] = None
    height: Optional[int] = None
    cost: StepCost = field(default_factory=StepCost)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMServingABC(ABC):

    name: str = ""
    model: str = ""

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        *,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ServingResult]:
        pass

    def generate_one(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> ServingResult:
        return self.generate([prompt], system=system, **kwargs)[0]


class VLMServingABC(ABC):

    name: str = ""
    model: str = ""

    @abstractmethod
    def score(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        rubric: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[ServingResult]:
        pass


class ImageGenServingABC(ABC):

    name: str = ""
    model: str = ""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageGenResult:
        pass

    @abstractmethod
    def edit(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mask_bytes: Optional[bytes] = None,
        size: Optional[str] = None,
        **kwargs: Any,
    ) -> ImageGenResult:
        pass
