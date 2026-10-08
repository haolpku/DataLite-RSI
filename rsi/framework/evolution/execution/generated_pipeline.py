"""Key-based adapter retained for generated pipelines authored against it.

The authored shape is now the reference-aligned
``OperatorABC``/``PipelineABC``/``FileStorage`` contract (see the provider
skills). This module stays so artifacts written against the earlier key-based
entry point keep running; new pipelines should not use it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from rsi.framework.core.contracts import TaskEnvelope
from rsi.framework import Operator, Pipeline, RunContext, RecordStore, StorageBundle

from rsi.framework.evolution.execution.step_cache import FrameworkStepCache


#: Entry formats the corpus loader recognizes, mirroring
#: ``rsi.framework.evolution.corpus.SUPPORTED_ENTRY_SUFFIXES``.
_TABULAR_SUFFIXES = {".json", ".csv", ".parquet"}


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Stream structured corpus rows.

    JSONL is streamed line by line so a large corpus never has to be held in
    memory. The other discoverable entry formats (``.json``, ``.csv``,
    ``.parquet``) have no line-oriented reading, so they are loaded through
    ``FileStorage``, which applies the same parsing and missing-value handling
    the rest of the runtime uses.
    """
    source = Path(path)
    if source.suffix.lower() in _TABULAR_SUFFIXES:
        from rsi.framework.io.dataflow_storage import FileStorage

        rows = FileStorage(first_entry_file_name=str(source)).step().read("dict")
        for number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"record {number} in {source.name} must be an object")
            yield row
        return
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {number} must be an object")
            yield row


def write_step_records(context: RunContext, rows: Iterable[Mapping[str, object]]) -> str:
    """Atomically write one stage through the shared RecordStore."""
    store = context.resources.get("step_store")
    index = context.resources.get("stage_index")
    if not isinstance(store, RecordStore) or not isinstance(index, int) or index < 1:
        raise RuntimeError("generated step requires framework RecordStore and stage_index")
    return str(store.write_jsonl(f"pipeline_step_step{index}", rows))


def run_generated_pipeline(
    operators: Iterable[Operator],
    *,
    raw_entry_path: str | Path,
    iteration_dir: str | Path,
) -> str | None:
    """Compile all stages, then execute the verified cache suffix on the shared runtime."""
    stages = tuple(operators)
    if not stages or any(not isinstance(stage, Operator) for stage in stages):
        raise ValueError("generated pipeline requires shared Operator instances")
    for index, stage in enumerate(stages):
        expected_input = "source_records" if index == 0 else stages[index - 1].output_keys[0]
        if len(stage.input_keys) != 1 or stage.input_keys[0] != expected_input:
            raise ValueError(f"{stage.name} must consume {expected_input!r}")
        if len(stage.output_keys) != 1:
            raise ValueError(f"{stage.name} must declare exactly one output key")
    result_key = stages[-1].output_keys[0]
    Pipeline("dataflow_generated", stages, result_keys=(result_key,)).compile(("source_records",))

    raw_path = Path(raw_entry_path).expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"fixed raw corpus is missing: {raw_path}")
    cache = FrameworkStepCache.from_env(
        raw_entry_path=str(raw_path),
        operator_names=[stage.__class__.__name__ for stage in stages],
    )
    if os.environ.get("DF_COMPILE_ONLY"):
        return None
    if cache.prefix_count == len(stages):
        return cache.entry_path

    root = Path(iteration_dir).resolve()
    input_key = "source_records" if not cache.prefix_count else stages[cache.prefix_count - 1].output_keys[0]
    suffix = stages[cache.prefix_count:]
    task = TaskEnvelope.from_mapping({
        "schema_version": "0.1",
        "task_id": root.name,
        "method_id": "dataflow-evolver",
        "objective": "Transform the fixed structured-record corpus",
        "data": {input_key: cache.entry_path},
        "output": {"required_keys": [result_key]},
        "workspace": str(root),
        "provider": "offline",
        "role": "pipeline_builder",
    })
    context = RunContext(
        task, "generated", StorageBundle(root, "generated"), resume=False,
        resources={"step_store": RecordStore(root / "cache"), "stage_offset": cache.prefix_count},
    )
    result = Pipeline("dataflow_generated_suffix", suffix, result_keys=(result_key,)).run(
        {input_key: cache.entry_path}, context
    )
    output_path = Path(result.output[result_key]).resolve()
    if not output_path.is_file() or not output_path.is_relative_to((root / "cache").resolve()):
        raise ValueError("generated operator returned a path outside the step RecordStore")
    return str(output_path)
