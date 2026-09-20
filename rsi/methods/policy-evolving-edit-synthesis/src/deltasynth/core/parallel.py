from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    def tqdm(it=None, **_kw):  # type: ignore
        return it if it is not None else iter(())


log = logging.getLogger("deltasynth")


def is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        return True
    msg = str(exc)
    if "HTTP 429" in msg:
        return True
    if re.search(r"\brate[_\s-]?limit", msg, flags=re.I):
        return True
    if re.search(r"\bquota\b", msg, flags=re.I):
        return True
    return False


def parse_retry_after(exc: BaseException) -> Optional[float]:
    if isinstance(exc, urllib.error.HTTPError):
        ra = exc.headers.get("Retry-After") if exc.headers else None
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return None


class RateLimiter:

    def __init__(self, max_concurrent: int = 8) -> None:
        self.max_concurrent = max_concurrent
        self._sem = threading.BoundedSemaphore(max_concurrent)

    def acquire(self) -> None:
        self._sem.acquire()

    def release(self) -> None:
        try:
            self._sem.release()
        except ValueError:
            pass

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


@dataclass(slots=True)
class FailureRecord:
    sample_id: str
    operator: str
    error_class: str
    error_message: str
    occurred_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FailureSink:

    FILE_NAME = "_failures.jsonl"

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / self.FILE_NAME
        self._lock = threading.Lock()

    def record(self, rec: FailureRecord) -> None:
        line = json.dumps(dataclasses.asdict(rec), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def list_failures(self) -> list[FailureRecord]:
        if not self.path.exists():
            return []
        out: list[FailureRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(FailureRecord(**d))
        return out

    def list_failed_sample_ids(self) -> set[str]:
        return {r.sample_id for r in self.list_failures()}

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def run_parallel(
    sample_ids: list[str],
    work_fn: Callable[[str], None],
    *,
    max_workers: int = 8,
    rate_limiter: Optional[RateLimiter] = None,
    failure_sink: Optional[FailureSink] = None,
    operator_name: str = "",
    progress_desc: Optional[str] = None,
    on_429_sleep_s: float = 30.0,
) -> tuple[list[str], list[FailureRecord]]:
    succeeded: list[str] = []
    failures: list[FailureRecord] = []

    if not sample_ids:
        return succeeded, failures

    desc = progress_desc or operator_name or "parallel"

    def _wrapped(sid: str) -> tuple[str, Optional[BaseException]]:
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            try:
                work_fn(sid)
                return sid, None
            except BaseException as exc:  # noqa: BLE001 — we route to FailureSink
                if is_rate_limited(exc):
                    cooldown = parse_retry_after(exc) or on_429_sleep_s
                    log.warning(
                        "[%s] 429 / rate-limit on %s -> cooling down %.1fs",
                        operator_name,
                        sid,
                        cooldown,
                    )
                    time.sleep(cooldown)
                return sid, exc
        finally:
            if rate_limiter is not None:
                rate_limiter.release()

    progress = tqdm(
        total=len(sample_ids),
        desc=desc,
        unit="sample",
        leave=True,
        dynamic_ncols=True,
    )
    fail_ct = 0
    succ_ct = 0

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"ds-{desc}") as pool:
        futs = [pool.submit(_wrapped, sid) for sid in sample_ids]
        for fut in as_completed(futs):
            sid, exc = fut.result()
            if exc is None:
                succeeded.append(sid)
                succ_ct += 1
            else:
                rec = FailureRecord(
                    sample_id=sid,
                    operator=operator_name,
                    error_class=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
                failures.append(rec)
                fail_ct += 1
                if failure_sink is not None:
                    failure_sink.record(rec)
                log.warning(
                    "[%s] sample %s FAILED: %s: %s",
                    operator_name,
                    sid,
                    type(exc).__name__,
                    str(exc)[:400],
                )
            progress.set_postfix(ok=succ_ct, fail=fail_ct, refresh=False)
            progress.update(1)
    progress.close()

    return succeeded, failures


class ParallelMixin:

    name: str = ""

    def run_parallel_samples(
        self,
        sample_ids: list[str],
        process_one: Callable[..., None],
        *,
        storage: Any,
        ctx: Any,
        max_workers: int = 8,
        progress_desc: Optional[str] = None,
    ) -> tuple[list[str], list[FailureRecord]]:
        rate_limiter = getattr(ctx, "rate_limiter", None)
        failure_sink = getattr(ctx, "failure_sink", None)

        if failure_sink is not None:
            already_failed = failure_sink.list_failed_sample_ids()
            sample_ids = [s for s in sample_ids if s not in already_failed]

        def _work(sid: str) -> None:
            process_one(storage, ctx, sid)

        return run_parallel(
            sample_ids,
            _work,
            max_workers=max_workers,
            rate_limiter=rate_limiter,
            failure_sink=failure_sink,
            operator_name=self.name,
            progress_desc=progress_desc or self.name,
        )


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
