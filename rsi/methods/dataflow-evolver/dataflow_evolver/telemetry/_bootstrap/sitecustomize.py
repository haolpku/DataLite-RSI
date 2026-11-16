"""Automatically installed only in generated pipeline subprocesses."""

import sys
from pathlib import Path

try:
    # The bootstrap directory itself is added to PYTHONPATH. Ensure the directory that
    # contains the dataflow_evolver package is importable even without an editable install.
    package_parent = str(Path(__file__).resolve().parents[3])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
except Exception as exc:
    raise SystemExit(f"DataFlow-Evolver bootstrap path setup failed: {exc}") from exc

try:
    from dataflow_evolver.compat.storage import install_file_storage_null_normalization

    install_file_storage_null_normalization()
except Exception as exc:
    # Storage normalization is a correctness boundary, not best-effort telemetry.
    raise SystemExit(f"DataFlow-Evolver storage compatibility setup failed: {exc}") from exc

try:
    from dataflow_evolver.compat.openai_serving import install_api_llm_reasoning_compat

    install_api_llm_reasoning_compat()
except Exception as exc:
    # Reasoning preservation is part of the generated dataset contract.
    raise SystemExit(f"DataFlow-Evolver API LLM compatibility setup failed: {exc}") from exc

try:
    from dataflow_evolver.telemetry.pipeline_usage import install_requests_usage_tracking

    install_requests_usage_tracking()
except Exception:
    # Telemetry is best-effort and must never prevent a pipeline from starting.
    pass
