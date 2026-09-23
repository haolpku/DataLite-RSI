from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight HLE evaluation")
    parser.add_argument("--config", default="/config/eval.json")
    parser.add_argument("--output")
    parser.add_argument("--sample-id")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("config must contain a JSON object")
    if args.output is not None:
        config["output"] = args.output
    if args.sample_id:
        config["sample_id"] = args.sample_id

    summary = evaluate(config)
    print(json.dumps({"summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
