"""Atomic, method-neutral record, blob, artifact and run storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.contracts import ArtifactRef, canonical_json


def _fs_path(path: Path) -> str:
    """Use extended Windows paths for nested run artifacts when needed."""
    value = str(path.absolute())
    if os.name == "nt" and len(value) >= 240 and not value.startswith("\\\\?\\"):
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value[2:]
        return "\\\\?\\" + value
    return value


def _relative_name(name: str) -> Path:
    path = Path(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} or ":" in part for part in path.parts
    ):
        raise ValueError(f"unsafe storage key {name!r}")
    return path


def atomic_bytes(path: Path, content: bytes) -> None:
    os.makedirs(_fs_path(path.parent), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".tmp", dir=_fs_path(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, _fs_path(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RecordStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._append_lock = threading.Lock()

    def _path(self, key: str, suffix: str) -> Path:
        return self.root / _relative_name(key + suffix)

    def write_json(self, key: str, value: Any) -> Path:
        path = self._path(key, ".json")
        atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        return path

    def read_json(self, key: str) -> Any:
        path = self._path(key, ".json")
        return json.loads(Path(_fs_path(path)).read_text(encoding="utf-8")) if os.path.isfile(_fs_path(path)) else None

    def write_jsonl(self, key: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        path = self._path(key, ".jsonl")
        os.makedirs(_fs_path(path.parent), exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".tmp", dir=_fs_path(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(canonical_json(dict(row)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, _fs_path(path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def read_jsonl(self, key: str) -> list[dict[str, Any]]:
        path = self._path(key, ".jsonl")
        if not os.path.isfile(_fs_path(path)):
            return []
        rows: list[dict[str, Any]] = []
        with open(_fs_path(path), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL row in {path} must be an object")
                    rows.append(value)
        return rows

    def append_jsonl(self, key: str, row: Mapping[str, Any]) -> Path:
        path = self._path(key, ".jsonl")
        os.makedirs(_fs_path(path.parent), exist_ok=True)
        with self._append_lock, open(_fs_path(path), "a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path


class BlobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put_bytes(self, content: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        path = self.root / digest[:2] / digest
        if not os.path.isfile(_fs_path(path)):
            atomic_bytes(path, content)
        return ArtifactRef("blob", path.relative_to(self.root.parent).as_posix(), media_type, digest)

    def put_file(self, source: str | Path, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        source_path = Path(source)
        digest = hashlib.sha256()
        with open(_fs_path(source_path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        path = self.root / digest.hexdigest()[:2] / digest.hexdigest()
        if not os.path.isfile(_fs_path(path)):
            os.makedirs(_fs_path(path.parent), exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".tmp", dir=_fs_path(path.parent))
            os.close(fd)
            try:
                shutil.copyfile(_fs_path(source_path), temporary)
                os.replace(temporary, _fs_path(path))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ArtifactRef("blob", path.relative_to(self.root.parent).as_posix(), media_type, digest.hexdigest())

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        if ref.kind not in {"blob", "image", "binary"}:
            raise ValueError("reference does not point to stored bytes")
        path = (self.root.parent / _relative_name(ref.uri)).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("blob reference escapes store")
        content = Path(_fs_path(path)).read_bytes()
        if ref.sha256 and hashlib.sha256(content).hexdigest() != ref.sha256:
            raise ValueError("blob checksum mismatch")
        return content


class ArtifactStore:
    def __init__(self, records: RecordStore, blobs: BlobStore) -> None:
        self.records = records
        self.blobs = blobs

    def put_image(self, name: str, image: bytes | str | Path, *, media_type: str = "image/png") -> ArtifactRef:
        blob = (
            self.blobs.put_bytes(image, media_type=media_type)
            if isinstance(image, bytes)
            else self.blobs.put_file(image, media_type=media_type)
        )
        ref = ArtifactRef("image", blob.uri, media_type, blob.sha256, {"name": name})
        self.records.write_json(f"artifacts/{name}", ref.to_dict())
        return ref

    def put_file(
        self, name: str, source: str | Path, *, kind: str = "binary",
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        """Import a generated file without reading a large media blob at once."""
        if kind not in {"image", "binary", "blob"}:
            raise ValueError("file artifact kind must be image, binary or blob")
        if kind == "image" and not media_type.startswith("image/"):
            raise ValueError("image artifact needs an image/* media_type")
        blob = self.blobs.put_file(source, media_type=media_type)
        ref = ArtifactRef(kind, blob.uri, media_type, blob.sha256, {**dict(metadata or {}), "name": name})
        self.records.write_json(f"artifacts/{name}", ref.to_dict())
        return ref

    def put_binary(self, name: str, data: bytes, *, media_type: str = "application/octet-stream") -> ArtifactRef:
        blob = self.blobs.put_bytes(data, media_type=media_type)
        ref = ArtifactRef("binary", blob.uri, media_type, blob.sha256, {"name": name})
        self.records.write_json(f"artifacts/{name}", ref.to_dict())
        return ref

    def put_video_reference(
        self, name: str, uri: str, *, metadata: Mapping[str, Any] | None = None,
        media_type: str = "video/*", sha256: str | None = None,
    ) -> ArtifactRef:
        if not uri:
            raise ValueError("video reference uri is required")
        if not media_type.startswith("video/"):
            raise ValueError("video reference needs a video/* media_type")
        ref = ArtifactRef("video_reference", uri, media_type, sha256,
                          metadata={**dict(metadata or {}), "name": name})
        self.records.write_json(f"artifacts/{name}", ref.to_dict())
        return ref


class RunStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.root = Path(run_dir)
        self.records = RecordStore(self.root)

    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        existing = self.records.read_json("run_manifest")
        if existing:
            identity = (
                "run_id", "task_id", "method_id", "pipeline_fingerprint",
                "input_fingerprint", "skill_fingerprints", "provider", "role",
                "input_contract", "diagnostic_isolation",
            )
            if any(existing.get(key) != payload.get(key) for key in identity):
                raise ValueError("run directory already contains a different run manifest")
        return self.records.write_json("run_manifest", dict(payload))

    def append_failure(self, payload: Mapping[str, Any]) -> Path:
        return self.records.append_jsonl("failures", payload)

    def write_observation(self, payload: Mapping[str, Any]) -> Path:
        return self.records.write_json("execution_observation", dict(payload))

    def write_diagnostic(self, name: str, payload: Mapping[str, Any]) -> Path:
        return self.records.write_json(f"diagnostics/{name}", dict(payload))


class StorageBundle:
    """One run's common stores; method artifacts can use any suitable store."""

    def __init__(self, workspace: str | Path, run_id: str) -> None:
        from ..core.checkpoint import CheckpointStore
        from ..core.provenance import ProvenanceStore

        safe_run_id = _relative_name(run_id)
        if len(safe_run_id.parts) != 1:
            raise ValueError("run_id must be one path component")
        self.root = Path(workspace).resolve() / "runs" / safe_run_id
        self.run_id = run_id
        self.records = RecordStore(self.root / "records")
        self.blobs = BlobStore(self.root / "blobs")
        self.artifacts = ArtifactStore(self.records, self.blobs)
        self.checkpoints = CheckpointStore(self.root / "checkpoints")
        self.provenance = ProvenanceStore(self.root / "provenance")
        self.run = RunStore(self.root)
