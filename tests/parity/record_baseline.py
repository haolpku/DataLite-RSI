"""Record the effective baseline from the real open-dataflow package.

Run this only inside an isolated environment that has ``open-dataflow==1.0.10``
installed, with the read-only DataFlow-Evolver checkout available for its
``compat`` fixes:

    conda run -n dfe-open-dataflow-parity python tests/parity/record_baseline.py

The recording is what the offline suite compares against, so it must come from
the reference implementation and never from ``rsi.framework``. Fixtures are
built in a temporary directory and only basenames and relative paths are
stored, so the file carries no machine-specific paths.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters  # noqa: E402
import cases  # noqa: E402


BASELINE_PATH = Path(__file__).resolve().parent / "baseline_open_dataflow.json"


def record() -> dict:
    import pandas

    import dataflow

    api = adapters.reference_api()
    with tempfile.TemporaryDirectory(prefix="odf-parity-") as tmp:
        results = cases.evaluate(api, Path(tmp))

    return {
        "environment": {
            "open_dataflow": getattr(dataflow, "__version__", "unknown"),
            "pandas": pandas.__version__,
            "python": platform.python_version(),
            "compat_source": "DataFlow-Evolver dataflow_evolver/compat",
        },
        "results": results,
    }


def main() -> int:
    payload = record()
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    groups = payload["results"]
    total = sum(len(group) for group in groups.values())
    print(f"wrote {BASELINE_PATH.name}: {total} cases across {len(groups)} groups")
    for name, group in sorted(groups.items()):
        failures = sum(1 for case in group.values() if not case["ok"])
        print(f"  {name}: {len(group)} cases, {failures} raising")
    print("environment:", json.dumps(payload["environment"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
