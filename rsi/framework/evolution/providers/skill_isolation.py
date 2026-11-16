"""Helpers for running diagnostic agents outside project skill discovery."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def isolated_agent_cwd(cwd: str, *, enabled: bool) -> Iterator[Path]:
    """Yield an empty cwd so backend project skill discovery cannot reach the repo."""
    if not enabled:
        yield Path(cwd)
        return
    with tempfile.TemporaryDirectory(prefix="dfe-agent-isolated-") as directory:
        yield Path(directory)
