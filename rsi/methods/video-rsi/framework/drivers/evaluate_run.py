"""CLI for fixed-policy, model-free evaluation of a VideoRSI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import EvaluationPolicy, evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-target-model", default="qwen3-vl-8b-instruct")
    args = parser.parse_args()
    result = evaluate_run(
        args.run_dir,
        EvaluationPolicy(frozen_target_model=args.frozen_target_model),
    )
    output = args.output or args.run_dir / "evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
