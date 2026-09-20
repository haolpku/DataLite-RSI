from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .context import RunContext
from .storage import DirStorage

log = logging.getLogger("deltasynth")


class OperatorABC(ABC):

    name: str = ""

    def __init__(self, *, name: Optional[str] = None) -> None:
        if name:
            self.name = name
        elif not self.name:
            self.name = self.__class__.__name__


    @abstractmethod
    def run(
        self,
        storage: DirStorage,
        ctx: RunContext,
        *,
        sample_ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        pass


    def _filter_pending(
        self,
        storage: DirStorage,
        sample_ids: list[str],
    ) -> list[str]:
        pending: list[str] = []
        for sid in sample_ids:
            if storage.is_step_completed(sid, self.name):
                log.debug("[%s] skipping completed sample %s", self.name, sid)
                continue
            pending.append(sid)
        return pending

    def _mark_done(self, storage: DirStorage, sample_id: str) -> None:
        storage.mark_step_completed(sample_id, self.name)


    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
