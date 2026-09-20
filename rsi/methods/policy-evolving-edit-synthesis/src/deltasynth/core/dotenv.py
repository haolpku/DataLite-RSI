from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _candidate_paths() -> Iterable[Path]:
    explicit = os.environ.get("DELTASYNTH_DOTENV")
    if explicit:
        yield Path(explicit)
    project_root = Path(__file__).resolve().parents[2]
    yield project_root / ".env"
    yield Path.cwd() / ".env"


def _parse_line(line: str) -> Optional[tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    k, v = line.split("=", 1)
    k = k.strip()
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {'"', "'"}:
        v = v[1:-1]
    return k, v


def load_dotenv(path: Optional[Path] = None, *, override: bool = False) -> int:
    paths: list[Path] = [path] if path else list(_candidate_paths())
    n_set = 0
    for p in paths:
        if not p or not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            kv = _parse_line(line)
            if kv is None:
                continue
            k, v = kv
            if not override and k in os.environ:
                continue
            os.environ[k] = v
            n_set += 1
        return n_set
    return n_set
