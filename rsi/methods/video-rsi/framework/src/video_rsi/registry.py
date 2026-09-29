"""Small explicit operator registry inspired by OpenDCAI/DataFlow."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class OperatorSpec:
    """Versioned, discoverable metadata for one reusable operator."""

    name: str
    version: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    task_types: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    description: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_keys"] = list(self.input_keys)
        value["output_keys"] = list(self.output_keys)
        value["task_types"] = list(self.task_types)
        value["capabilities"] = list(self.capabilities)
        value["ref"] = self.ref
        return value


class OperatorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, type] = {}

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            key = name or cls.__name__
            if key in self._items and self._items[key] is not cls:
                raise KeyError(f"operator {key!r} is already registered")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown operator {name!r}; available={sorted(self._items)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def spec(self, name: str) -> OperatorSpec:
        cls = self.get(name)
        return OperatorSpec(
            name=name,
            version=str(getattr(cls, "version", "1")),
            input_keys=tuple(getattr(cls, "input_keys", ())),
            output_keys=tuple(getattr(cls, "output_keys", ())),
            task_types=tuple(getattr(cls, "task_types", ())),
            capabilities=tuple(getattr(cls, "capabilities", ())),
            description=str(getattr(cls, "description", "")),
        )

    def specs(self, task_type: str | None = None) -> list[OperatorSpec]:
        specs = [self.spec(name) for name in self.names()]
        if task_type is not None:
            specs = [s for s in specs if not s.task_types or task_type in s.task_types]
        return specs

    def resolve(self, ref: str, **kwargs: Any) -> Any:
        """Instantiate ``Name`` or ``Name@version`` with compatibility checks."""
        name, _, requested_version = ref.partition("@")
        cls = self.get(name)
        actual_version = str(getattr(cls, "version", "1"))
        if requested_version and requested_version != actual_version:
            raise KeyError(
                f"operator {ref!r} is not available; registered version is {actual_version!r}"
            )
        return cls(**kwargs)


OPERATOR_REGISTRY = OperatorRegistry()
