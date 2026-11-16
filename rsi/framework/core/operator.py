"""Method-neutral operator contract and stable stage specification."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .contracts import fingerprint

if TYPE_CHECKING:
    from .pipeline import RunContext


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    version: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    config: Mapping[str, Any]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "input_keys": list(self.input_keys),
            "output_keys": list(self.output_keys),
            "config": dict(self.config),
            "fingerprint": self.fingerprint,
        }


class Operator(ABC):
    name = "operator"
    version = "1"
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()

    def config(self) -> dict[str, Any]:
        return {}

    @property
    def fingerprint(self) -> str:
        try:
            source = inspect.getsource(self.__class__)
        except (OSError, TypeError):
            source = self.__class__.__qualname__
        return fingerprint(
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "source": source,
                "name": self.name,
                "version": self.version,
                "config": self.config(),
            }
        )

    @abstractmethod
    def run(self, state: dict[str, Any], context: "RunContext") -> Mapping[str, Any]:
        raise NotImplementedError

    def stats(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: len(value) if isinstance(value, (list, dict)) else 1
            for key, value in outputs.items()
        }
