"""Model-serving contract injected into DataFlow operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMServingABC(ABC):
    """Abstract model/API service used by a generated pipeline operator."""

    @abstractmethod
    def generate_from_input(
        self,
        user_inputs: list[str],
        system_prompt: str = "You are a helpful assistant",
    ) -> list[str | None]:
        """Return one response aligned with each input."""

    @abstractmethod
    def start_serving(self) -> None:
        """Start a locally managed service when necessary."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release transport and model resources."""

    def load_model(self, model_name_or_path: str, **kwargs: Any):
        """Optional model-loading hook for serving implementations."""
        raise NotImplementedError("This method should be implemented by subclasses.")
