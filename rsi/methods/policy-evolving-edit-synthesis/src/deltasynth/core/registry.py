from __future__ import annotations

import importlib
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}
        self._lazy_modules: list[str] = []
        self._lazy_loaded = False


    def register(self, name: Optional[str] = None) -> Callable[[type[T]], type[T]]:

        def deco(cls: type[T]) -> type[T]:
            key = name or cls.__name__
            if key in self._items and self._items[key] is not cls:
                raise ValueError(
                    f"{self._kind} '{key}' is already registered to {self._items[key]!r}"
                )
            self._items[key] = cls
            return cls

        return deco

    def register_class(self, name: str, cls: type[T]) -> None:
        self.register(name)(cls)


    def add_lazy_module(self, module_path: str) -> None:
        if module_path not in self._lazy_modules:
            self._lazy_modules.append(module_path)

    def _maybe_load_lazy(self) -> None:
        if self._lazy_loaded:
            return
        self._lazy_loaded = True
        for m in self._lazy_modules:
            try:
                importlib.import_module(m)
            except Exception as exc:  # noqa: BLE001 — surface but don't crash
                import logging

                logging.getLogger("deltasynth").warning(
                    "Lazy import of %s failed: %s", m, exc
                )


    def get(self, name: str) -> type[T]:
        self._maybe_load_lazy()
        if name not in self._items:
            raise KeyError(
                f"{self._kind} '{name}' not registered. Known: {sorted(self._items)}"
            )
        return self._items[name]

    def has(self, name: str) -> bool:
        self._maybe_load_lazy()
        return name in self._items

    def list(self) -> list[str]:
        self._maybe_load_lazy()
        return sorted(self._items)


    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        self._maybe_load_lazy()
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry({self._kind!r}, n={len(self._items)})"


OPERATOR_REGISTRY: Registry = Registry("operator")
SERVING_REGISTRY: Registry = Registry("serving")
PIPELINE_REGISTRY: Registry = Registry("pipeline")
