"""Make the in-tree package and driver modules importable under pytest.

The framework follows a ``src/`` layout, while the repository-wide checker
invokes pytest from the DataLite-RSI root.  Keeping this adjustment local to the
test suite avoids requiring an editable install just to run deterministic tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = FRAMEWORK_ROOT / "src"

for path in (str(FRAMEWORK_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
