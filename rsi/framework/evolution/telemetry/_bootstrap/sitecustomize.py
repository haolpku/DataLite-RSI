"""Automatically installed only in generated pipeline subprocesses."""

import sys
from pathlib import Path

try:
    # The bootstrap directory itself is added to PYTHONPATH. Ensure the
    # repository package remains importable without an editable install.
    package_parent = str(Path(__file__).resolve().parents[5])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
except Exception as exc:
    raise SystemExit(f"pipeline telemetry bootstrap path setup failed: {exc}") from exc

try:
    from rsi.framework.evolution.telemetry.pipeline_usage import install_requests_usage_tracking

    install_requests_usage_tracking()
except Exception:
    # Telemetry is best-effort and must never prevent a pipeline from starting.
    pass
