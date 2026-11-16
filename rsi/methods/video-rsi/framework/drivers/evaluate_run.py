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
    parser.add_argument(
        "--evaluation-model", "--frozen-target-model", dest="evaluation_model",
        help="Expected evaluator model ID; omit to infer it from the run manifest.",
    )
    parser.add_argument("--model-role", choices=("target", "reference"))
    args = parser.parse_args()
    policy = None
    if args.evaluation_model or args.model_role:
        policy = EvaluationPolicy(
            frozen_target_model=args.evaluation_model,
            model_role=args.model_role or "reference",
        )
    result = evaluate_run(args.run_dir, policy)
    output = args.output or args.run_dir / "evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
