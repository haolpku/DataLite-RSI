"""Operator and serving contracts aligned with ``open-dataflow`` 1.0.10.

Generated operators subclass :class:`OperatorABC` and receive a storage view
per call; a pipeline injects an :class:`LLMServingABC` implementation into any
operator that needs model access. ``PipelineABC.compile()`` discovers operator
members by their :class:`OperatorABC` type and ``_build_operator_nodes_graph``
finds an operator's serving by scanning its attributes for
:class:`LLMServingABC`, so both base classes must be the types generated code
actually inherits from.

Not reproduced: the reference's ``OPERATOR_REGISTRY``/``get_operator`` lookup
and its ``ALLOWED_PROMPTS`` prompt library. Generated pipelines instantiate
local operator classes directly and never resolve one by registry name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .runtime_logger import get_logger


class LLMServingABC(ABC):
    """Model access handed to an operator by its pipeline."""

    @abstractmethod
    def generate_from_input(
        self, user_inputs: List[str], system_prompt: str
    ) -> List[str]:
        """Return one aligned response per input."""

    @abstractmethod
    def start_serving(self) -> None:
        """Bring the backing service up, if it is locally managed."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release transport and any held compute resources."""

    def load_model(self, model_name_or_path: str, **kwargs: Any):
        raise NotImplementedError("This method should be implemented by subclasses.")


class OperatorABC(ABC):
    """One coherent processing step over a storage view."""

    def __init__(self) -> None:
        self.logger = get_logger()

    @abstractmethod
    def run(self) -> None:
        """Main function to run the operator."""
