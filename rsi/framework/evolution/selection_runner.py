"""Execute one registered selection policy in a separate Python process."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types


def main() -> None:
    context = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    policy = context["policy"]
    path = Path(policy["policy_file"])
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != policy["sha256"]:
        raise ValueError("selection policy changed; start a new run")
    sys.path.insert(0, str(path.parent))
    module = types.ModuleType("_datalite_selection_policy")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    select = getattr(module, "select", None)
    if not callable(select):
        raise ValueError("selection policy must define select(context)")
    result = select(context)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
