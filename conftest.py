"""Keep independent benchmark evaluator modules isolated during collection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    path = Path(str(module_path))
    if path.name == "test_evaluator.py" and "benchmarks" in path.parts:
        sys.modules.pop("evaluator", None)
