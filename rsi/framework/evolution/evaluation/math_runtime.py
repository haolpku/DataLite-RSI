from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import yaml

PROTOCOL_VERSION = 3


class RuntimeConfigError(ValueError):
    """Raised when the external runtime configuration is invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeConfigError(f"configuration root must be an object: {path}")
    return _expand_environment(raw)


def load_request(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeConfigError("request root must be a JSON object")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeConfigError(
            f"unsupported request protocol_version={raw.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    stage = raw.get("stage")
    if stage not in {"baseline", "periodic"}:
        raise RuntimeConfigError(f"unsupported request stage: {stage!r}")
    training_performed = raw.get("training_performed")
    if not isinstance(training_performed, bool):
        raise RuntimeConfigError("training_performed must be boolean")
    if stage == "baseline":
        if training_performed:
            raise RuntimeConfigError(
                "baseline request must set training_performed=false"
            )
    else:
        if not training_performed:
            raise RuntimeConfigError(
                "periodic request must set training_performed=true"
            )
        dataset_path = raw.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise RuntimeConfigError("periodic request must provide dataset_path")
    benchmarks = raw.get("expected_benchmarks")
    if (
        not isinstance(benchmarks, list)
        or not benchmarks
        or not all(isinstance(item, str) and item.strip() for item in benchmarks)
    ):
        raise RuntimeConfigError("expected_benchmarks must be a non-empty string list")
    bad_cases = raw.get("bad_cases_per_benchmark")
    if isinstance(bad_cases, bool) or not isinstance(bad_cases, int) or bad_cases < 1:
        raise RuntimeConfigError("bad_cases_per_benchmark must be a positive integer")
    return raw


def normalize_sft_dataset(
    source: Path,
    destination: Path,
    *,
    user_field: str,
    assistant_field: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    skipped_blank = 0
    with (
        source.open(encoding="utf-8") as input_handle,
        destination.open("w", encoding="utf-8", newline="\n") as output_handle,
    ):
        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise RuntimeConfigError(
                    f"dataset line {line_number} is not a JSON object"
                )
            user = record.get(user_field)
            assistant = record.get(assistant_field)
            if not isinstance(user, str) or not user.strip():
                skipped_blank += 1
                continue
            if not isinstance(assistant, str) or not assistant.strip():
                skipped_blank += 1
                continue
            normalized = {"instruction": user, "input": "", "output": assistant}
            output_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows += 1
    if rows == 0:
        raise RuntimeConfigError("normalized SFT dataset is empty")
    return {
        "source_path": str(source.resolve()),
        "normalized_path": str(destination.resolve()),
        "rows": rows,
        "skipped_blank_rows": skipped_blank,
        "user_field": user_field,
        "assistant_field": assistant_field,
    }


def audit_sft_token_lengths(
    dataset_path: Path,
    *,
    model_path: str,
    cutoff_len: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lengths: list[int] = []
    for record in _read_jsonl(dataset_path):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": record["output"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
    ordered = sorted(lengths)
    total_tokens = sum(ordered)
    retained_tokens = sum(min(length, cutoff_len) for length in ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "cutoff_len": cutoff_len,
        "rows_over_cutoff": sum(length > cutoff_len for length in ordered),
        "retained_token_fraction": retained_tokens / total_tokens
        if total_tokens
        else 1.0,
    }


def synchronize_model_eos_contract(model_dir: Path) -> dict[str, Any]:
    """Align model generation EOS with the tokenizer used by the SFT template."""
    tokenizer_path = model_dir / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        raise RuntimeError(
            f"LlamaFactory completed without tokenizer_config.json: {model_dir}"
        )
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, Mapping):
        eos_token = eos_token.get("content")
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError(f"saved tokenizer has no usable eos_token: {tokenizer_path}")

    eos_token_id = tokenizer_config.get("eos_token_id")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        eos_token_id = None
        decoder = tokenizer_config.get("added_tokens_decoder", {})
        if isinstance(decoder, Mapping):
            for raw_token_id, token_spec in decoder.items():
                content = (
                    token_spec.get("content")
                    if isinstance(token_spec, Mapping)
                    else token_spec
                )
                if content == eos_token:
                    try:
                        eos_token_id = int(raw_token_id)
                    except (TypeError, ValueError):
                        continue
                    break
    if eos_token_id is None:
        added_tokens_path = model_dir / "added_tokens.json"
        if added_tokens_path.is_file():
            added_tokens = json.loads(added_tokens_path.read_text(encoding="utf-8"))
            candidate = added_tokens.get(eos_token)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                eos_token_id = candidate
    if eos_token_id is None:
        raise RuntimeError(
            f"cannot resolve token id for saved tokenizer EOS {eos_token!r}"
        )

    report: dict[str, Any] = {
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "configs": {},
    }
    for file_name in ("config.json", "generation_config.json"):
        config_path = model_dir / file_name
        if not config_path.is_file():
            if file_name == "config.json":
                raise RuntimeError(
                    f"LlamaFactory completed without a model config: {model_dir}"
                )
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        previous = payload.get("eos_token_id")
        payload["eos_token_id"] = eos_token_id
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["configs"][file_name] = {
            "previous_eos_token_id": previous,
            "eos_token_id": eos_token_id,
        }
    return report


def train_sft(
    request: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_dir: Path,
) -> tuple[Path, dict[str, str]]:
    sft = _mapping(config.get("sft"), "sft")
    source = Path(str(request["dataset_path"])).resolve()
    if not source.is_file():
        raise RuntimeConfigError(f"SFT dataset does not exist: {source}")

    dataset_dir = runtime_dir / "sft_dataset"
    train_file = dataset_dir / "train.jsonl"
    normalization = normalize_sft_dataset(
        source,
        train_file,
        user_field=_string(sft.get("user_field", "instruction"), "sft.user_field"),
        assistant_field=_string(
            sft.get("assistant_field", "output"), "sft.assistant_field"
        ),
    )
    cutoff_len = _positive_int(
        _mapping(sft.get("trainer"), "sft.trainer").get("cutoff_len"),
        "sft.trainer.cutoff_len",
    )
    normalization["token_lengths"] = audit_sft_token_lengths(
        train_file,
        model_path=_string(config.get("base_model"), "base_model"),
        cutoff_len=cutoff_len,
    )
    if (
        bool(sft.get("fail_on_truncation", False))
        and normalization["token_lengths"]["rows_over_cutoff"]
    ):
        raise RuntimeConfigError(
            f"{normalization['token_lengths']['rows_over_cutoff']} rows exceed cutoff_len={cutoff_len}"
        )
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "downstream_sft": {
                    "file_name": train_file.name,
                    "columns": {
                        "prompt": "instruction",
                        "query": "input",
                        "response": "output",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    normalization_path = runtime_dir / "normalization_report.json"
    normalization_path.write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_dir = runtime_dir / "sft_model"
    trainer = dict(_mapping(sft.get("trainer"), "sft.trainer"))
    required = {
        "model_name_or_path": _string(config.get("base_model"), "base_model"),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset": "downstream_sft",
        "output_dir": str(output_dir.resolve()),
    }
    forbidden_overrides = sorted(set(required).intersection(trainer))
    if forbidden_overrides:
        raise RuntimeConfigError(
            "sft.trainer must not override runtime-owned keys: "
            + ", ".join(forbidden_overrides)
        )
    trainer_config = {**required, **trainer}
    train_config_path = runtime_dir / "llamafactory_sft.yaml"
    train_config_path.write_text(
        yaml.safe_dump(trainer_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    cli = _string(sft.get("llamafactory_cli"), "sft.llamafactory_cli")
    env = os.environ.copy()
    _prepend_executable_dir(env, cli)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": _string(
                sft.get("cuda_visible_devices", "0,1,2,3,4,5,6,7"),
                "sft.cuda_visible_devices",
            ),
            "FORCE_TORCHRUN": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.update(_string_mapping(sft.get("env", {}), "sft.env"))
    run_logged(
        [cli, "train", str(train_config_path.resolve())],
        cwd=runtime_dir,
        env=env,
        stdout_path=runtime_dir / "sft_stdout.log",
        stderr_path=runtime_dir / "sft_stderr.log",
        timeout_sec=_positive_int(sft.get("timeout_sec", 86400), "sft.timeout_sec"),
    )
    if not (output_dir / "config.json").is_file():
        raise RuntimeError(
            f"LlamaFactory completed without a model config: {output_dir}"
        )
    eos_contract = synchronize_model_eos_contract(output_dir)
    expected_eos_token = sft.get("expected_eos_token")
    if expected_eos_token is not None:
        expected_token = _string(
            expected_eos_token, "sft.expected_eos_token"
        )
        if eos_contract["eos_token"] != expected_token:
            raise RuntimeConfigError(
                "SFT tokenizer EOS contract mismatch: "
                f"expected token {expected_token!r}, "
                f"got {eos_contract['eos_token']!r}"
            )
    expected_eos_token_id = sft.get("expected_eos_token_id")
    if expected_eos_token_id is not None:
        expected_token_id = _nonnegative_int(
            expected_eos_token_id, "sft.expected_eos_token_id"
        )
        if eos_contract["eos_token_id"] != expected_token_id:
            raise RuntimeConfigError(
                "SFT tokenizer EOS contract mismatch: "
                f"expected token id {expected_token_id}, "
                f"got {eos_contract['eos_token_id']}"
            )
    eos_contract_path = runtime_dir / "sft_eos_contract.json"
    eos_contract_path.write_text(
        json.dumps(eos_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir, {
        "checkpoint": str(output_dir.resolve()),
        "normalized_dataset": str(train_file.resolve()),
        "normalization_report": str(normalization_path.resolve()),
        "sft_config": str(train_config_path.resolve()),
        "sft_eos_contract": str(eos_contract_path.resolve()),
    }


def evaluate_model(
    model_path: Path,
    expected_benchmarks: Iterable[str],
    bad_cases_per_benchmark: int,
    config: Mapping[str, Any],
    runtime_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    groups = evaluation.get("groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeConfigError("evaluation.groups must be a non-empty list")

    benchmark_specs: dict[str, tuple[Mapping[str, Any], str]] = {}
    for group in groups:
        group_map = _mapping(group, "evaluation.groups[]")
        mapping = _mapping(
            group_map.get("benchmarks"), "evaluation.groups[].benchmarks"
        )
        for label, parser_name in mapping.items():
            if label in benchmark_specs:
                raise RuntimeConfigError(
                    f"duplicate benchmark label in evaluation groups: {label}"
                )
            benchmark_specs[str(label)] = (
                group_map,
                _string(parser_name, f"parser name for {label}"),
            )

    expected = list(expected_benchmarks)
    if set(expected) != set(benchmark_specs):
        raise RuntimeConfigError(
            "request/config benchmark mismatch: "
            f"request={sorted(expected)}, config={sorted(benchmark_specs)}"
        )

    evaluation_root = Path(__file__).resolve().parent / "vendor/qwen25_math"
    script = evaluation_root / "math_eval.py"
    if not script.is_file():
        raise RuntimeConfigError(f"vendored Qwen2.5-Math evaluator not found: {script}")
    python = _string(evaluation.get("python"), "evaluation.python")
    global_cuda = _string(
        evaluation.get("cuda_visible_devices", "0"),
        "evaluation.cuda_visible_devices",
    )
    timeout_sec = _positive_int(
        evaluation.get("timeout_sec", 86400), "evaluation.timeout_sec"
    )
    configured_cache_root = Path(
        _string(
            evaluation.get(
                "local_cache_root", str(Path(tempfile.gettempdir()) / "dataflow-evolver-vllm-evaluation")
            ),
            "evaluation.local_cache_root",
        )
    ).expanduser()
    local_cache_root = configured_cache_root.resolve()
    try:
        local_cache_root.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError as exc:
        raise RuntimeConfigError(
            "evaluation.local_cache_root must resolve under the system temporary directory so vLLM compile "
            f"caches stay on node-local storage: {local_cache_root}"
        ) from exc
    evaluation_env = _string_mapping(
        evaluation.get("env", {}), "evaluation.env"
    )
    process_cache_root = local_cache_root / f"runtime_{os.getpid()}"

    results: dict[str, Any] = {}
    output_root = runtime_dir / "evaluation"
    for group_index, group in enumerate(groups):
        group_map = _mapping(group, "evaluation.groups[]")
        group_name = _string(
            group_map.get("name", f"group_{group_index}"), "evaluation group name"
        )
        labels_to_parsers = {
            str(label): _string(parser_name, f"parser name for {label}")
            for label, parser_name in _mapping(
                group_map.get("benchmarks"), "evaluation.groups[].benchmarks"
            ).items()
        }
        data_dir = Path(_string(group_map.get("data_dir"), f"{group_name}.data_dir"))
        if not data_dir.is_dir():
            raise RuntimeConfigError(f"diagnostic data_dir does not exist: {data_dir}")
        group_output = output_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        parser_names = list(labels_to_parsers.values())
        prompt_type = _string(
            group_map.get("prompt_type", "qwen25-math-cot"),
            f"{group_name}.prompt_type",
        )
        apply_chat_template = bool(group_map.get("apply_chat_template", False))
        if prompt_type == "qwen25-math-cot" and apply_chat_template:
            raise RuntimeConfigError(
                "qwen25-math-cot already contains complete Qwen ChatML markers; "
                "evaluation.groups[].apply_chat_template must remain false"
            )
        cuda_visible_devices = _string(
            group_map.get("cuda_visible_devices", global_cuda),
            f"{group_name}.cuda_visible_devices",
        )
        data_parallel_size = _positive_int(
            group_map.get(
                "data_parallel_size", evaluation.get("data_parallel_size", 1)
            ),
            f"{group_name}.data_parallel_size",
        )
        tensor_parallel_size = _positive_int(
            group_map.get(
                "tensor_parallel_size", evaluation.get("tensor_parallel_size", 1)
            ),
            f"{group_name}.tensor_parallel_size",
        )
        pipeline_parallel_size = _positive_int(
            group_map.get(
                "pipeline_parallel_size",
                evaluation.get("pipeline_parallel_size", 1),
            ),
            f"{group_name}.pipeline_parallel_size",
        )
        visible_devices = _validate_evaluation_parallelism(
            cuda_visible_devices,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            field_name=f"{group_name}.cuda_visible_devices",
        )
        argv = [
            python,
            str(script),
            "--data_names",
            ",".join(parser_names),
            "--data_dir",
            str(data_dir.resolve()),
            "--model_name_or_path",
            str(model_path.resolve()),
            "--prompt_type",
            prompt_type,
            "--split",
            "test",
            "--num_test_sample",
            "-1",
            "--seed",
            str(_nonnegative_int(group_map.get("seed", 0), f"{group_name}.seed")),
            "--temperature",
            str(
                _number(group_map.get("temperature", 0.0), f"{group_name}.temperature")
            ),
            "--n_sampling",
            str(
                _positive_int(
                    group_map.get("n_sampling", 1), f"{group_name}.n_sampling"
                )
            ),
            "--top_p",
            str(_number(group_map.get("top_p", 1.0), f"{group_name}.top_p")),
            "--max_tokens_per_call",
            str(
                _positive_int(
                    group_map.get("max_tokens_per_call", 16384),
                    f"{group_name}.max_tokens_per_call",
                )
            ),
            "--max_model_len",
            str(
                _positive_int(
                    group_map.get("max_model_len", 32768),
                    f"{group_name}.max_model_len",
                )
            ),
            "--data_parallel_size",
            str(data_parallel_size),
            "--tensor_parallel_size",
            str(tensor_parallel_size),
            "--pipeline_parallel_size",
            str(pipeline_parallel_size),
            "--use_vllm",
            "--save_outputs",
            "--overwrite",
        ]
        if apply_chat_template:
            argv.append("--apply_chat_template")
        devices_per_replica = tensor_parallel_size * pipeline_parallel_size
        shard_outputs: list[Path] = []
        with ThreadPoolExecutor(max_workers=data_parallel_size) as executor:
            futures = []
            for rank in range(data_parallel_size):
                shard_output = group_output / f"shard_{rank:03d}"
                shard_output.mkdir(parents=True, exist_ok=True)
                shard_outputs.append(shard_output)
                start = rank * devices_per_replica
                rank_devices = visible_devices[start : start + devices_per_replica]
                rank_env = os.environ.copy()
                rank_env.update(evaluation_env)
                rank_env["CUDA_VISIBLE_DEVICES"] = ",".join(rank_devices)
                rank_env["TOKENIZERS_PARALLELISM"] = "false"
                rank_cache_root = (
                    process_cache_root
                    / f"group_{group_index:03d}"
                    / f"rank_{rank:03d}"
                )
                rank_cache_dirs = {
                    "XDG_CACHE_HOME": rank_cache_root / "xdg",
                    "VLLM_CACHE_ROOT": rank_cache_root / "vllm",
                    "TORCHINDUCTOR_CACHE_DIR": rank_cache_root / "torchinductor",
                    "TRITON_CACHE_DIR": rank_cache_root / "triton",
                    "CUDA_CACHE_PATH": rank_cache_root / "cuda",
                    "TORCH_EXTENSIONS_DIR": rank_cache_root / "torch_extensions",
                }
                for cache_dir in rank_cache_dirs.values():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                rank_env.update(
                    {name: str(path) for name, path in rank_cache_dirs.items()}
                )
                rank_argv = [
                    *argv,
                    "--output_dir",
                    str(shard_output.resolve()),
                    "--data_parallel_rank",
                    str(rank),
                ]
                futures.append(
                    executor.submit(
                        run_logged,
                        rank_argv,
                        cwd=evaluation_root,
                        env=rank_env,
                        stdout_path=shard_output / "stdout.log",
                        stderr_path=shard_output / "stderr.log",
                        timeout_sec=timeout_sec,
                    )
                )
            for future in futures:
                future.result()
        metric = _string(group_map.get("metric", "accuracy"), f"{group_name}.metric")
        n_sampling = _positive_int(
            group_map.get("n_sampling", 1), f"{group_name}.n_sampling"
        )
        for label, parser_name in labels_to_parsers.items():
            records_path = group_output / parser_name / "merged.jsonl"
            _merge_evaluation_shards(
                shard_outputs,
                parser_name=parser_name,
                destination=records_path,
            )
            records = _read_jsonl(records_path)
            results[label] = summarize_qwen_records(
                records,
                metric=metric,
                n_sampling=n_sampling,
                bad_cases_per_benchmark=bad_cases_per_benchmark,
                max_tokens_per_call=_positive_int(
                    group_map.get("max_tokens_per_call", 16384),
                    f"{group_name}.max_tokens_per_call",
                ),
            )

    return {label: results[label] for label in expected}, {
        "evaluation_output": str(output_root.resolve()),
        "evaluation_config": str(
            Path(_string(config.get("config_path"), "config_path")).resolve()
        ),
        "diagnostic_manifest": str(
            Path(
                _string(
                    evaluation.get("diagnostic_manifest"),
                    "evaluation.diagnostic_manifest",
                )
            ).resolve()
        ),
    }


def summarize_qwen_records(
    records: list[dict[str, Any]],
    *,
    metric: str,
    n_sampling: int,
    bad_cases_per_benchmark: int,
    max_tokens_per_call: int = 16384,
) -> dict[str, Any]:
    if not records:
        raise RuntimeConfigError("Qwen evaluator produced an empty result file")
    correctness: list[bool] = []
    failures: list[dict[str, Any]] = []

    def append_failure(
        record: dict[str, Any],
        sample_index: int,
        outputs: list[Any],
        parsed_answers: list[Any],
        finish_reasons: list[Any],
        output_token_counts: list[Any],
    ) -> None:
        question = _render_text(record.get("question"), "question")
        reference = _render_text(record.get("gt"), "ground truth")
        response = _render_text(
            outputs[sample_index],
            "model response",
            empty_sentinel="[empty model response]",
        )
        parsed_answer = _render_text(
            parsed_answers[sample_index],
            "parsed answer",
            empty_sentinel="",
        )
        finish_reason = str(finish_reasons[sample_index] or "")
        raw_token_count = output_token_counts[sample_index]
        response_token_count = (
            int(raw_token_count) if raw_token_count is not None else None
        )
        length_truncated = finish_reason == "length" or (
            response_token_count is not None
            and response_token_count >= max_tokens_per_call - 1
        )
        failures.append(
            {
                "question": question,
                "model_response": response,
                "parsed_answer": parsed_answer,
                "correct_answer": reference,
                "response_token_count": response_token_count,
                "finish_reason": finish_reason,
                "length_truncated": length_truncated,
                "sample_index": sample_index,
            }
        )

    for index, record in enumerate(records):
        scores = record.get("score")
        outputs = record.get("code")
        parsed_answers = record.get("pred")
        finish_reasons = record.get("finish_reason")
        output_token_counts = record.get("output_token_count")
        if not isinstance(scores, list) or len(scores) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has {len(scores) if isinstance(scores, list) else 'invalid'} "
                f"scores; expected {n_sampling}"
            )
        if not isinstance(outputs, list) or len(outputs) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid generated outputs"
            )
        if not isinstance(parsed_answers, list) or len(parsed_answers) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid parsed answers"
            )
        if finish_reasons is None:
            finish_reasons = [""] * n_sampling
        if not isinstance(finish_reasons, list) or len(finish_reasons) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid finish reasons"
            )
        if output_token_counts is None:
            output_token_counts = [None] * n_sampling
        if not isinstance(output_token_counts, list) or len(output_token_counts) != n_sampling:
            raise RuntimeConfigError(
                f"Qwen result {index} has invalid output token counts"
            )

        sample_correctness = [bool(score) for score in scores]
        if metric == "accuracy":
            correct = sample_correctness[0]
            correctness.append(correct)
            if not correct:
                append_failure(
                    record, 0, outputs, parsed_answers, finish_reasons, output_token_counts
                )
        elif metric == f"pass@{n_sampling}":
            correct = any(sample_correctness)
            correctness.append(correct)
            if not correct:
                append_failure(
                    record, 0, outputs, parsed_answers, finish_reasons, output_token_counts
                )
        elif metric == f"avg@{n_sampling}":
            correctness.extend(sample_correctness)
            for sample_index, correct in enumerate(sample_correctness):
                if not correct:
                    append_failure(
                        record,
                        sample_index,
                        outputs,
                        parsed_answers,
                        finish_reasons,
                        output_token_counts,
                    )
        else:
            raise RuntimeConfigError(
                f"unsupported metric {metric!r}; use 'accuracy', "
                f"'pass@{n_sampling}', or 'avg@{n_sampling}'"
            )

    correct_count = sum(correctness)
    total_count = len(correctness)
    failure_counts = _count_bad_case_categories(failures)
    bad_cases = _select_stratified_bad_cases(
        failures,
        limit=bad_cases_per_benchmark,
    )
    return {
        "score": correct_count / total_count,
        "correct_count": correct_count,
        "total_count": total_count,
        "question_count": len(records),
        "n_sampling": n_sampling,
        "metric": metric,
        "incorrect_count": len(failures),
        **failure_counts,
        "bad_cases": bad_cases,
    }

def _bad_case_category(case: dict[str, Any]) -> str:
    finish_reason = str(case.get("finish_reason") or "")
    if bool(case.get("length_truncated")) or finish_reason not in {"", "stop"}:
        return "runaway"
    if not str(case.get("parsed_answer") or "").strip():
        return "parse_failure"
    return "wrong_answer"


def _count_bad_case_categories(
    failures: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {
        "wrong_answer_count": 0,
        "parse_failure_count": 0,
        "runaway_count": 0,
    }
    for case in failures:
        counts[f"{_bad_case_category(case)}_count"] += 1
    return counts


def _select_stratified_bad_cases(
    failures: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Randomly cover categories by priority, then fill remaining slots."""
    if limit <= 0 or not failures:
        return []

    buckets = {
        category: [
            case for case in failures if _bad_case_category(case) == category
        ]
        for category in ("wrong_answer", "parse_failure", "runaway")
    }
    rng = random.Random(42)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def take(category: str, count: int) -> None:
        for case in buckets[category]:
            if len(selected) >= limit or count <= 0:
                return
            if id(case) not in selected_ids:
                selected.append(case)
                selected_ids.add(id(case))
                count -= 1

    for category in ("wrong_answer", "parse_failure", "runaway"):
        take(category, 1)
    for category in ("wrong_answer", "parse_failure", "runaway"):
        take(category, limit - len(selected))
    return selected


def run_logged(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: int,
) -> float:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with code {completed.returncode}; see {stdout_path} and {stderr_path}"
        )
    return elapsed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeConfigError(f"non-object JSONL record in {path}")
                records.append(value)
    return records


def _single_output_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.jsonl"))
    if len(candidates) != 1:
        raise RuntimeConfigError(
            f"expected one Qwen output JSONL under {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _merge_evaluation_shards(
    shard_outputs: Iterable[Path], *, parser_name: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output_handle:
        for shard_output in shard_outputs:
            shard_path = _single_output_jsonl(shard_output / parser_name)
            with shard_path.open(encoding="utf-8") as input_handle:
                for line in input_handle:
                    if line.strip():
                        output_handle.write(line.rstrip("\n") + "\n")


def _percentile(ordered: list[int], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _prepend_executable_dir(env: dict[str, str], executable: str) -> None:
    """Keep console-script subprocesses in the same Python environment."""
    executable_dir = str(Path(executable).resolve().parent)
    current_path = env.get("PATH", "")
    env["PATH"] = (
        os.pathsep.join((executable_dir, current_path))
        if current_path
        else executable_dir
    )


def _validate_evaluation_parallelism(
    cuda_visible_devices: str,
    *,
    data_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    field_name: str,
) -> list[str]:
    devices = [device.strip() for device in cuda_visible_devices.split(",")]
    if any(not device for device in devices):
        raise RuntimeConfigError(f"{field_name} contains an empty GPU identifier")
    if len(set(devices)) != len(devices):
        raise RuntimeConfigError(f"{field_name} contains duplicate GPU identifiers")
    required = (
        data_parallel_size * tensor_parallel_size * pipeline_parallel_size
    )
    if len(devices) != required:
        raise RuntimeConfigError(
            f"{field_name} exposes {len(devices)} GPUs, but evaluation parallelism "
            f"requires data_parallel_size * tensor_parallel_size * "
            f"pipeline_parallel_size = {required}"
        )
    return devices


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{field_name} must be an object")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeConfigError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeConfigError(f"{field_name} must be a non-negative integer")
    return value


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigError(f"{field_name} must be numeric")
    return float(value)


def _string_mapping(value: Any, field_name: str) -> dict[str, str]:
    mapping = _mapping(value, field_name)
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in mapping.items()
    ):
        raise RuntimeConfigError(f"{field_name} must map strings to strings")
    return dict(mapping)


def _render_text(value: Any, field_name: str, empty_sentinel: str | None = None) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    else:
        text = json.dumps(value, ensure_ascii=False)
    if not text.strip():
        if empty_sentinel is not None:
            return empty_sentinel
        raise RuntimeConfigError(f"Qwen result has empty {field_name}")
    return text
