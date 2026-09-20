from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from .context import RunContext
from .operator import OperatorABC
from .storage import DirStorage

log = logging.getLogger("deltasynth")


class PipelineABC(ABC):

    name: str = ""
    PIPELINE_STATE_NAME = "_pipeline_state.json"

    def __init__(
        self,
        storage: DirStorage,
        ctx: RunContext,
        *,
        name: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.ctx = ctx
        if name:
            self.name = name
        elif not self.name:
            self.name = self.__class__.__name__
        self.steps: list[OperatorABC] = list(self.build())


    @abstractmethod
    def build(self) -> list[OperatorABC]:
        pass


    def forward(self, **inputs: Any) -> list[str]:
        ids: Optional[list[str]] = None
        first = True
        for op in self.steps:
            if not first and ids == []:
                log.info(
                    "[pipeline=%s] stopping before %s; upstream returned 0 samples",
                    self.name,
                    op.name,
                )
                break
            kwargs = inputs if first else {}
            log.info("[pipeline=%s] running %s", self.name, op.name)
            ids = op.run(self.storage.step(), self.ctx, sample_ids=ids, **kwargs)
            self._mark_pipeline_step(op.name, count=len(ids or []))
            first = False
        return ids or []

    def run(self, **inputs: Any) -> list[str]:
        log.info(
            "[pipeline=%s] start run_id=%s storage=%s",
            self.name,
            self.ctx.run_id,
            self.storage.root,
        )
        out = self.forward(**inputs)
        log.info("[pipeline=%s] done; %d samples touched", self.name, len(out))
        return out


    @property
    def state_path(self) -> Path:
        return self.storage.root / self.PIPELINE_STATE_NAME

    def _read_pipeline_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _mark_pipeline_step(self, op_name: str, *, count: int) -> None:
        state = self._read_pipeline_state()
        runs = state.setdefault(self.ctx.run_id, {"pipeline": self.name, "steps": []})
        runs["steps"].append({"op": op_name, "samples": count})
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
