from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar

from .schema import CostMeta, StepCost

T = TypeVar("T")
log = logging.getLogger("deltasynth")


class CostTracker:

    def __init__(self) -> None:
        self.totals = CostMeta()
        self.by_sample: dict[str, CostMeta] = {}
        self._lock = threading.Lock()

    def add(self, sample_id: str, step: StepCost) -> None:
        with self._lock:
            self.totals.add(step)
            bucket = self.by_sample.setdefault(sample_id, CostMeta())
            bucket.add(step)

    def get(self, sample_id: str) -> CostMeta:
        with self._lock:
            return self.by_sample.setdefault(sample_id, CostMeta())

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "samples": len(self.by_sample),
                "api_calls": len(self.totals.steps),
                "total_tokens": self.totals.total_tokens,
                "latency_total_s": self.totals.latency_total_s,
            }


PERMANENT_ERROR_MARKERS: tuple[str, ...] = (
    "invalid_api_key",
    "insufficient_quota",
    "http 401",
    "http 403",
)

STOCHASTIC_ERROR_MARKERS: tuple[str, ...] = (
    "moderation_blocked",
    "rejected by the safety system",
    "content_policy_violation",
    "image_generation_user_error",
)


def is_permanent_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in PERMANENT_ERROR_MARKERS)


def is_stochastic_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in STOCHASTIC_ERROR_MARKERS)


@dataclass(slots=True)
class RetryPolicy:

    max_retries: int = 3
    backoff_s: float = 2.0
    backoff_factor: float = 2.0
    backoff_cap_s: float = 8.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def call(
        self,
        fn: Callable[..., T],
        *args: Any,
        op_name: str = "<call>",
        **kwargs: Any,
    ) -> T:
        attempt = 0
        delay = self.backoff_s
        last_exc: Optional[BaseException] = None
        while attempt <= self.max_retries:
            try:
                return fn(*args, **kwargs)
            except self.retry_on as exc:  # noqa: BLE001 — caller picks scope
                last_exc = exc
                if is_permanent_failure(exc):
                    log.warning(
                        "[%s] permanent failure, not retrying: %s",
                        op_name,
                        str(exc)[:300],
                    )
                    break
                if attempt == self.max_retries:
                    break
                wait = 0.0 if is_stochastic_failure(exc) else delay
                log.warning(
                    "[%s] attempt %d/%d failed: %s -> retrying in %.1fs",
                    op_name,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                if wait:
                    time.sleep(wait)
                    delay = min(delay * self.backoff_factor, self.backoff_cap_s)
                attempt += 1
        assert last_exc is not None
        raise last_exc


@dataclass(slots=True)
class RunContext:

    run_id: str = field(default_factory=lambda: _default_run_id())
    seed: int = 0
    prng: random.Random = field(default_factory=random.Random)
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    env: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    notes: dict[str, Any] = field(default_factory=dict)
    rate_limiter: Optional[Any] = None
    failure_sink: Optional[Any] = None
    max_workers: int = 1

    def __post_init__(self) -> None:
        self.prng.seed(self.seed)


    def env_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        if key in self.env:
            return self.env[key]
        return os.environ.get(key, default)

    def env_require(self, key: str) -> str:
        v = self.env_get(key)
        if v is None or v == "":
            raise KeyError(f"Required env var '{key}' is not set")
        return v


    def call_with_retry(
        self,
        fn: Callable[..., T],
        *args: Any,
        op_name: str = "<call>",
        **kwargs: Any,
    ) -> T:
        return self.retry_policy.call(fn, *args, op_name=op_name, **kwargs)


def _default_run_id() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
