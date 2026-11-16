"""Operator contract aligned with ``open-dataflow`` 1.0.10.

Generated operators subclass :class:`OperatorABC` and receive a storage view
per call. The serving contract lives in :mod:`llm_serving`; the pipeline
imports that type separately when it scans operator members for injected model
services.

Not reproduced: the reference's ``OPERATOR_REGISTRY``/``get_operator`` lookup
and its ``ALLOWED_PROMPTS`` prompt library. Generated pipelines instantiate
local operator classes directly and never resolve one by registry name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .runtime_logger import get_logger


class OperatorABC(ABC):
    """One coherent processing step over a storage view."""

    def __init__(self) -> None:
        self.logger = get_logger()

    @abstractmethod
    def run(self) -> None:
        """Main function to run the operator."""
