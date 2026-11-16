"""Resolve source media referenced by a fixed structured input corpus."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..core.contracts import ArtifactRef

if TYPE_CHECKING:
    from ..io.storage import ArtifactStore


def artifact_store_for(iteration_dir: str | Path) -> "ArtifactStore":
    """Return the artifact store a generated pipeline writes media into.

    **DataLite extension, not part of the open-dataflow contract.** The
    reference runtime has no artifact concept; this exists so an operator that
    produces images or video references can persist them without inventing its
    own layout.

    A ``run(self, storage)`` operator has no ``RunContext``, so this resolves
    the same store the key-based runtime would have supplied: stored blobs get
    a ``blobs/...`` URI relative to ``{iteration_dir}/runs/generated``, which is
    what the outer loop's reference rehoming and the step cache's
    ``contains_local_blob_refs`` check already expect.
    """
    from ..io.storage import StorageBundle

    return StorageBundle(iteration_dir, "generated").artifacts


def resolve_local_input_artifact(
    entry_path: str | Path, reference: ArtifactRef | Mapping[str, Any]
) -> Path:
    """Resolve a local source artifact relative to its JSONL entry manifest."""
    ref = reference if isinstance(reference, ArtifactRef) else ArtifactRef.from_mapping(reference)
    if "://" in ref.uri:
        raise ValueError("remote input artifact requires a task-specific backend")
    path = Path(ref.uri).expanduser()
    resolved = (path if path.is_absolute() else Path(entry_path).resolve().parent / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input artifact is missing: {resolved}")
    return resolved
