from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .schema import Sample

try:
    import fcntl  # type: ignore

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

PathLike = Union[str, Path]


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_FCNTL:
        yield
        return
    f = open(lock_path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # type: ignore[union-attr]
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        except Exception:
            pass
        f.close()


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_FCNTL:
        yield
        return
    f = open(lock_path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # type: ignore[union-attr]
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        except Exception:
            pass
        f.close()


class DirStorage:

    METADATA_NAME = "metadata.json"
    STATE_NAME = "_state.json"
    INDEX_NAME = "_index.jsonl"

    def __init__(self, root: PathLike) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._step_idx = 0
        self._index_lock = threading.Lock()
        self._state_locks: dict[str, threading.Lock] = {}
        self._state_locks_master = threading.Lock()

    def _state_lock_for(self, sample_id: str) -> threading.Lock:
        with self._state_locks_master:
            lk = self._state_locks.get(sample_id)
            if lk is None:
                lk = threading.Lock()
                self._state_locks[sample_id] = lk
            return lk

    @property
    def _index_lock_path(self) -> Path:
        return self.root / "._index.lock"


    def sample_dir(self, sample_id: str) -> Path:
        return self.root / sample_id

    def metadata_path(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / self.METADATA_NAME

    def state_path(self, sample_id: str) -> Path:
        return self.sample_dir(sample_id) / self.STATE_NAME

    @property
    def index_path(self) -> Path:
        return self.root / self.INDEX_NAME


    def write_sample(self, sample: Sample, *, update_index: bool = True) -> Path:
        d = self.sample_dir(sample.sample_id)
        d.mkdir(parents=True, exist_ok=True)
        with self._state_lock_for(sample.sample_id):
            _atomic_write_text(
                self.metadata_path(sample.sample_id),
                sample.model_dump_json(exclude_none=True, indent=2),
            )
        if update_index:
            self._index_upsert(
                {
                    "sample_id": sample.sample_id,
                    "dir": sample.sample_id,
                    "task_type": sample.task_type.value,
                    "category": sample.category,
                    "subcategory": sample.subcategory,
                }
            )
        return self.metadata_path(sample.sample_id)

    def read_sample(self, sample_id: str) -> Sample:
        path = self.metadata_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(f"No metadata.json at {path}")
        return Sample.model_validate_json(path.read_text(encoding="utf-8"))

    def has_sample(self, sample_id: str) -> bool:
        return self.metadata_path(sample_id).exists()

    def delete_sample(self, sample_id: str) -> None:
        d = self.sample_dir(sample_id)
        if d.exists():
            shutil.rmtree(d)
        self._index_drop(sample_id)

    def list_samples(self) -> Iterator[str]:
        for child in self.root.iterdir():
            if child.is_dir() and (child / self.METADATA_NAME).exists():
                yield child.name

    def iter_samples(self) -> Iterator[Sample]:
        for sid in self.list_samples():
            yield self.read_sample(sid)


    def artifact_path(self, sample_id: str, name: str) -> Path:
        return self.sample_dir(sample_id) / name

    def write_artifact(
        self,
        sample_id: str,
        name: str,
        data: Union[bytes, PathLike],
    ) -> Path:
        target = self.artifact_path(sample_id, name)
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (bytes, bytearray)):
            _atomic_write_bytes(target, bytes(data))
        else:
            src = Path(data)
            if not src.exists():
                raise FileNotFoundError(f"Source artifact not found: {src}")
            shutil.copyfile(src, target)
        return target

    def read_artifact(self, sample_id: str, name: str) -> Path:
        p = self.artifact_path(sample_id, name)
        if not p.exists():
            raise FileNotFoundError(f"Artifact not found: {p}")
        return p


    def write_state(self, sample_id: str, state: dict[str, Any]) -> None:
        d = self.sample_dir(sample_id)
        d.mkdir(parents=True, exist_ok=True)
        with self._state_lock_for(sample_id):
            _atomic_write_text(
                self.state_path(sample_id),
                json.dumps(state, ensure_ascii=False, indent=2),
            )

    def read_state(self, sample_id: str) -> dict[str, Any]:
        p = self.state_path(sample_id)
        if not p.exists():
            return {}
        with self._state_lock_for(sample_id):
            return json.loads(p.read_text(encoding="utf-8"))

    def mark_step_completed(self, sample_id: str, step_name: str) -> None:
        with self._state_lock_for(sample_id):
            p = self.state_path(sample_id)
            state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            completed = list(state.get("completed", []))
            if step_name not in completed:
                completed.append(step_name)
            state["completed"] = completed
            self.sample_dir(sample_id).mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                p, json.dumps(state, ensure_ascii=False, indent=2)
            )

    def is_step_completed(self, sample_id: str, step_name: str) -> bool:
        return step_name in self.read_state(sample_id).get("completed", [])


    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        if body:
            body += "\n"
        _atomic_write_text(self.index_path, body)

    def _index_upsert(self, row: dict[str, Any]) -> None:
        with self._index_lock, _file_lock(self._index_lock_path):
            rows = self._read_index()
            sid = row["sample_id"]
            replaced = False
            for i, r in enumerate(rows):
                if r.get("sample_id") == sid:
                    rows[i] = row
                    replaced = True
                    break
            if not replaced:
                rows.append(row)
            self._write_index(rows)

    def _index_drop(self, sample_id: str) -> None:
        with self._index_lock, _file_lock(self._index_lock_path):
            rows = [r for r in self._read_index() if r.get("sample_id") != sample_id]
            self._write_index(rows)

    def reindex(self) -> int:
        rows: list[dict[str, Any]] = []
        for sid in sorted(self.list_samples()):
            try:
                s = self.read_sample(sid)
                rows.append(
                    {
                        "sample_id": s.sample_id,
                        "dir": s.sample_id,
                        "task_type": s.task_type.value,
                        "category": s.category,
                        "subcategory": s.subcategory,
                    }
                )
            except Exception:  # noqa: BLE001 — skip broken samples on reindex
                continue
        self._write_index(rows)
        return len(rows)

    def index_rows(self) -> list[dict[str, Any]]:
        return self._read_index()


    def step(self) -> "DirStorage":
        self._step_idx += 1
        return self

    @property
    def cursor(self) -> int:
        return self._step_idx

    @contextmanager
    def at_cursor(self, idx: int) -> Iterator["DirStorage"]:
        prev = self._step_idx
        self._step_idx = idx
        try:
            yield self
        finally:
            self._step_idx = prev


    def __repr__(self) -> str:  # pragma: no cover
        return f"DirStorage(root={self.root!s}, cursor={self._step_idx})"
