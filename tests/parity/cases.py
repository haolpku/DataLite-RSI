"""Shared parity cases for the dataflow-compatible runtime.

Each case is a fixture plus a probe that is written once and evaluated twice:
against the real ``open-dataflow==1.0.10`` package (plus the DataFlow-Evolver
``compat`` fixes, which together form the effective baseline), and against
``rsi.framework``. The two runs must agree.

The probes receive their implementation through an ``api`` bundle, so nothing
here imports either implementation. ``record_baseline.py`` evaluates them
against the reference and stores the result; the offline suite evaluates them
against ``rsi.framework`` and compares.

Results must be JSON-serializable and must not contain absolute paths: probes
return path *basenames* and relative paths only, so a recording made on one
machine is comparable on another.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable


# --------------------------------------------------------------------------
# helpers shared by probes
# --------------------------------------------------------------------------

def _describe(value: Any) -> Any:
    """Describe a value as [type_name, value] so coercion differences show."""
    if isinstance(value, dict):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_describe(item) for item in value]
    name = type(value).__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return [name, repr(value)]
        return [name, value]
    return [name, str(value)]


def _normalize_message(message: str) -> str:
    """Strip machine- and implementation-specific text from an error message.

    Error *text* is not the behavior under comparison: the fixture's temporary
    directory and the serving class's own name legitimately differ between a
    recording and a later run. Replacing them keeps the comparison on the
    exception type and the part of the message that describes the failure.
    """
    normalized = re.sub(r"[A-Za-z]:\\\\[^\s\"']+|[A-Za-z]:\\[^\s\"']+|/tmp/[^\s\"']+", "<path>", message)
    normalized = normalized.replace("APILLMServing_request", "<serving>")
    normalized = normalized.replace("PipelineLLMServing", "<serving>")
    # Operator fixtures are defined inside these probes, so their repr carries
    # the probe's own qualname and a heap address.
    normalized = re.sub(r"<[\w.<>]*Op object at 0x[0-9a-fA-F]+>", "<operator>", normalized)
    return normalized


def _outcome(fn: Callable[[], Any]) -> dict[str, Any]:
    """Capture a probe result or its exception type and message."""
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:  # noqa: BLE001 - the error itself is the observation
        return {"ok": False, "error": _normalize_message(f"{type(exc).__name__}: {exc}")}


def _basename(path: Any) -> Any:
    return os.path.basename(str(path)) if path is not None else None


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# group 1: storage read/write, type coercion, errors
# --------------------------------------------------------------------------

#: JSONL fixtures exercising pandas inference and missing-value handling.
READ_FIXTURES: dict[str, str] = {
    "int_with_null": '{"a":1}\n{"a":null}\n',
    "int_only": '{"a":1}\n{"a":2}\n',
    "bool_with_null": '{"a":true}\n{"a":null}\n',
    "bool_only": '{"a":true}\n{"a":false}\n',
    "str_with_null": '{"a":"x"}\n{"a":null}\n',
    "mixed_int_str": '{"a":1}\n{"a":"x"}\n',
    "numeric_string": '{"a":"0123"}\n{"a":"456"}\n',
    "numeric_string_spaced": '{"a":" 12"}\n{"a":"3"}\n',
    "exponent_string": '{"a":"1e3"}\n{"a":"2"}\n',
    "underscore_string": '{"a":"1_000"}\n',
    "hex_string": '{"a":"0x10"}\n',
    "date_column": '{"date":"2024-01-02"}\n{"date":"2024-02-03"}\n',
    "plain_date_string": '{"a":"2024-01-02"}\n{"a":"2024-02-03"}\n',
    "nested_objects": '{"a":{"k":1},"b":[1,2]}\n{"a":{"k":2},"b":[]}\n',
    "nested_inner_null": '{"a":{"k":null},"b":[1,null]}\n',
    "ragged_rows": '{"a":1}\n{"a":2,"b":"x"}\n',
    "all_null_column": '{"a":null}\n{"a":null}\n',
    "empty_file": "",
    "blank_lines": '{"a":1}\n\n{"a":2}\n',
    "zero_and_null": '{"a":0}\n{"a":null}\n',
    "empty_string_and_null": '{"a":""}\n{"a":null}\n',
    "unicode": '{"a":"caf\\u00e9 \\u2713"}\n',
    "key_order": '{"z":1,"a":2,"m":3}\n',
    "row_key_order_differs": '{"z":1,"a":2}\n{"a":3,"z":4}\n',
    "huge_int": '{"a":9223372036854775807}\n',
    "over_int64": '{"a":9223372036854775808}\n',
    "oversized_int": '{"a":123456789012345678901234567890}\n',
    "bool_and_numeric_string": '{"a":true}\n{"a":"2"}\n',
    "nan_inf_strings": '{"a":"NaN"}\n{"a":"inf"}\n',
}

#: Row payloads exercising write-side frame construction and serialization.
WRITE_FIXTURES: dict[str, Any] = {
    "none_value": [{"a": None, "b": "x"}],
    "ragged": [{"a": 1}, {"b": 2}],
    "ragged_str": [{"a": "x"}, {"b": "y"}],
    "nested_null": [{"a": {"k": None}, "b": [1, None]}],
    "int_and_none": [{"a": 1}, {"a": None}, {"a": 3}],
    "float_and_none": [{"a": 1.5}, {"a": None}],
    "bool_and_none": [{"a": True}, {"a": None}],
    "bool_and_int": [{"a": True}, {"a": 2}],
    "str_and_int": [{"a": "s"}, {"a": 1}],
    "numeric_string": [{"a": "0123"}],
    "key_order": [{"z": 1, "a": 2}],
    "row_key_order_differs": [{"z": 1, "a": 2}, {"a": 3, "z": 4}],
    "new_key_later": [{"a": 1}, {"a": 2, "b": 3}],
    "nan_float": [{"a": float("nan")}],
    "inf_float": [{"a": float("inf")}],
    "empty_dicts": [{}, {}],
    "none_only": [{"a": None}],
    "unicode": [{"a": "café ✓"}],
    "surrogate": [{"t": "ok\udc00bad", "n": 3, "o": {"k": [1, None]}}],
    "tuple_value": [{"a": (1, 2)}],
    "empty_list": [],
    "scalar_list": [1, 2],
    "plain_dict": {"a": 1},
}


def storage_cases(api, root: Path) -> dict[str, Any]:
    """Probe FileStorage reads, writes, step numbering and failure modes."""
    FileStorage = api["FileStorage"]
    out: dict[str, Any] = {}

    for name, text in READ_FIXTURES.items():
        path = _write(root / "read" / f"{name}.jsonl", text)
        out[f"read/{name}"] = _outcome(
            lambda p=path: _describe(FileStorage(first_entry_file_name=str(p)).step().read("dict"))
        )

    seed = _write(root / "seed.jsonl", '{"s":1}\n')
    for name, rows in WRITE_FIXTURES.items():
        cache = root / "write" / name

        def write_probe(rows=rows, cache=cache):
            storage = FileStorage(
                first_entry_file_name=str(seed), cache_path=str(cache)
            ).step()
            path = storage.write(rows)
            return {
                "file": _basename(path),
                "content": Path(path).read_text(encoding="utf-8"),
            }

        out[f"write/{name}"] = _outcome(write_probe)

    # step numbering and the copy semantics of step()
    def step_semantics():
        entry = _write(root / "steps.jsonl", '{"a":1,"b":null}\n{"a":2,"b":"t"}\n')
        storage = FileStorage(
            first_entry_file_name=str(entry),
            cache_path=str(root / "steps"),
            file_name_prefix="pfx",
            cache_type="jsonl",
        )
        initial = storage.operator_step
        view = storage.step()
        return {
            "initial_operator_step": initial,
            "parent_after_step": storage.operator_step,
            "view_is_parent": view is storage,
            "view_step": view.operator_step,
            "first_write": _basename(view.write([{"a": 1}])),
            "second_step": storage.step().operator_step,
            "third_write": _basename(storage.step().write([{"a": 1}])),
            "reset": storage.reset().operator_step,
            "keys": view.get_keys_from_dataframe(),
        }

    out["step/semantics"] = _outcome(step_semantics)

    # read/write before step() advances the counter
    out["step/read_before_step"] = _outcome(
        lambda: FileStorage(first_entry_file_name=str(seed)).read("dict")
    )
    out["step/missing_file"] = _outcome(
        lambda: FileStorage(first_entry_file_name=str(root / "absent.jsonl")).step().read("dict")
    )
    out["step/empty_entry_name_dict"] = _outcome(
        lambda: FileStorage(first_entry_file_name="").step().read("dict")
    )
    out["step/empty_entry_name_shape"] = _outcome(
        lambda: list(FileStorage(first_entry_file_name="").step().read().shape)
    )
    out["step/unsupported_output_type"] = _outcome(
        lambda: FileStorage(first_entry_file_name=str(seed)).step().read("pickle")
    )

    # file naming across prefixes and cache types
    def naming(prefix: str, cache_type: str, suffix: str):
        entry = _write(root / f"name_{suffix}.jsonl", '{"a":1}\n')
        storage = FileStorage(
            first_entry_file_name=str(entry),
            cache_path=str(root / f"name_{suffix}"),
            file_name_prefix=prefix,
            cache_type=cache_type,
        )
        view = storage.step()
        written = view.write([{"a": 1, "b": None}])
        return {"file": _basename(written), "roundtrip": _describe(storage.step().read("dict"))}

    out["naming/default_prefix_jsonl"] = _outcome(
        lambda: naming("dataflow_cache_step", "jsonl", "a")
    )
    out["naming/custom_prefix_csv"] = _outcome(lambda: naming("pipeline_step", "csv", "b"))
    out["naming/custom_prefix_json"] = _outcome(lambda: naming("pipeline_step", "json", "c"))

    # the entry format comes from the file extension, not cache_type
    for ext, text in (
        ("json", json.dumps([{"a": 1, "b": None}])),
        ("csv", "a,b\n1,\n2,t\n"),
    ):
        path = _write(root / f"entry.{ext}", text)
        out[f"entry_format/{ext}"] = _outcome(
            lambda p=path: _describe(
                FileStorage(first_entry_file_name=str(p)).step().read("dict")
            )
        )
    dotted = _write(root / "my.data.jsonl", '{"a":1}\n')
    out["entry_format/dotted_name"] = _outcome(
        lambda: _describe(FileStorage(first_entry_file_name=str(dotted)).step().read("dict"))
    )

    # a multi-step chain, which is what a generated pipeline actually produces
    def chain():
        entry = _write(root / "chain.jsonl", '{"a":1}\n{"a":2}\n')
        storage = FileStorage(
            first_entry_file_name=str(entry),
            cache_path=str(root / "chain"),
            file_name_prefix="pipeline_step",
            cache_type="jsonl",
        )
        files = []
        first = storage.step()
        files.append(_basename(first.write([{**r, "s1": True} for r in first.read("dict")])))
        second = storage.step()
        files.append(_basename(second.write([{**r, "s2": True} for r in second.read("dict")])))
        final = Path(root / "chain" / "pipeline_step_step2.jsonl")
        return {"files": files, "final": final.read_text(encoding="utf-8")}

    out["chain/two_steps"] = _outcome(chain)
    return out


# --------------------------------------------------------------------------
# group 2: compile / forward trace and graph
# --------------------------------------------------------------------------

def pipeline_cases(api, root: Path) -> dict[str, Any]:
    """Probe compile/forward ordering, key validation and serving lifecycle."""
    OperatorABC = api["OperatorABC"]
    PipelineABC = api["PipelineABC"]
    FileStorage = api["FileStorage"]
    LLMServingABC = api["LLMServingABC"]
    out: dict[str, Any] = {}

    def build(tag: str, *, nops=2, side_effect=False, call_twice=False,
              no_storage=False, extra_kwargs=None, serving=False):
        events: list[Any] = []
        entry = _write(root / f"{tag}.jsonl", '{"a":1}\n{"a":2}\n')

        class Op(OperatorABC):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def run(self, storage, **kwargs):
                events.append(["run", self.index, dict(kwargs)])
                rows = storage.read("dict")
                storage.write(rows)

        class Serving(LLMServingABC):
            def __init__(self):
                self.cleanups = 0

            def generate_from_input(self, user_inputs, system_prompt="s"):
                return [None] * len(user_inputs)

            def start_serving(self):
                pass

            def cleanup(self):
                self.cleanups += 1
                events.append(["cleanup"])

        class Pipe(PipelineABC):
            def __init__(self):
                super().__init__()
                self.storage = FileStorage(
                    first_entry_file_name=str(entry),
                    cache_path=str(root / tag),
                    file_name_prefix="pipeline_step",
                    cache_type="jsonl",
                )
                self.serving = Serving() if serving else None
                for index in range(nops):
                    operator = Op(index)
                    if serving:
                        operator.llm_serving = self.serving
                    setattr(self, f"op{index}", operator)

            def forward(self):
                if side_effect:
                    events.append(["side-effect"])
                for index in range(nops):
                    operator = getattr(self, f"op{index}")
                    kwargs = dict(extra_kwargs or {})
                    if no_storage:
                        operator.run(**kwargs)
                    else:
                        operator.run(storage=self.storage.step(), **kwargs)
                    if call_twice:
                        operator.run(storage=self.storage.step(), **kwargs)

        return Pipe, events

    # compile runs forward() once: non-operator side effects fire, bodies do not
    def compile_phase():
        Pipe, events = build("compile_phase", side_effect=True)
        pipeline = Pipe()
        pipeline.compile()
        cache = root / "compile_phase"
        return {
            "events_during_compile": list(events),
            "cache_files_after_compile": sorted(p.name for p in cache.glob("*")) if cache.exists() else [],
            "operator_step_after_compile": pipeline.storage.operator_step,
            "recorded_operators": [r.op_name for r in pipeline.op_runtimes],
            "accumulated_keys": pipeline.accumulated_keys,
            "final_keys": pipeline.final_keys,
            "node_names": [n.op_name for n in pipeline.op_nodes_list],
            "last_modified_index_of_keys": pipeline.last_modified_index_of_keys,
        }

    out["compile/phase"] = _outcome(compile_phase)

    def forward_phase():
        Pipe, events = build("forward_phase", side_effect=True)
        pipeline = Pipe()
        pipeline.compile()
        events.clear()
        pipeline.forward()
        cache = root / "forward_phase"
        return {
            "events_during_forward": list(events),
            "cache_files": sorted(p.name for p in cache.glob("*")),
        }

    out["compile/forward_events"] = _outcome(forward_phase)

    def called_twice():
        Pipe, events = build("twice", nops=1, call_twice=True)
        pipeline = Pipe()
        pipeline.compile()
        recorded = [r.op_name for r in pipeline.op_runtimes]
        events.clear()
        pipeline.forward()
        return {"recorded": recorded, "runs": [e for e in events if e[0] == "run"]}

    out["compile/operator_called_twice"] = _outcome(called_twice)

    out["compile/missing_storage_kwarg"] = _outcome(
        lambda: build("no_storage", nops=1, no_storage=True)[0]().compile()
    )

    def extra_kwargs_case():
        Pipe, events = build("extra_kw", nops=1, extra_kwargs={"threshold": 0.5})
        pipeline = Pipe()
        pipeline.compile()
        recorded = [dict(r.kwargs) for r in pipeline.op_runtimes]
        events.clear()
        pipeline.forward()
        return {"recorded": recorded, "runs": [e for e in events if e[0] == "run"]}

    out["compile/extra_kwargs_forwarded"] = _outcome(extra_kwargs_case)

    def compile_twice():
        Pipe, _ = build("compile_twice", nops=1)
        pipeline = Pipe()
        pipeline.compile()
        return pipeline.compile()

    out["compile/twice_raises"] = _outcome(compile_twice)

    def forward_without_compile():
        Pipe, events = build("no_compile", nops=1)
        Pipe().forward()
        return [e for e in events if e[0] == "run"]

    out["compile/forward_without_compile"] = _outcome(forward_without_compile)

    def zero_operators():
        class Empty(PipelineABC):
            def forward(self):
                pass

        pipeline = Empty()
        pipeline.compile()
        return {
            "recorded": [r.op_name for r in pipeline.op_runtimes],
            "accumulated_keys": pipeline.accumulated_keys,
            "node_names": [n.op_name for n in pipeline.op_nodes_list],
        }

    out["compile/zero_operators"] = _outcome(zero_operators)

    # input_*/output_* arguments drive compile-time key validation
    def key_validation(input_key: str, output_key: str, tag: str):
        entry = _write(root / f"keys_{tag}.jsonl", '{"a":1}\n')

        class KeyedOp(OperatorABC):
            def __init__(self):
                super().__init__()

            def run(self, storage, input_key=None, output_key=None):
                pass

        class Pipe(PipelineABC):
            def __init__(self):
                super().__init__()
                self.storage = FileStorage(
                    first_entry_file_name=str(entry), cache_path=str(root / f"keys_{tag}")
                )
                self.op = KeyedOp()

            def forward(self):
                self.op.run(
                    storage=self.storage.step(),
                    input_key=input_key,
                    output_key=output_key,
                )

        pipeline = Pipe()
        pipeline.compile()
        return {
            "accumulated_keys": pipeline.accumulated_keys,
            "final_keys": pipeline.final_keys,
        }

    out["keys/valid"] = _outcome(lambda: key_validation("a", "b", "ok"))
    out["keys/unknown_input"] = _outcome(lambda: key_validation("nope", "b", "bad"))

    def serving_lifecycle():
        Pipe, events = build("serving", nops=2, serving=True)
        pipeline = Pipe()
        pipeline.compile()
        counter_before = sum(pipeline.llm_serving_counter.values())
        events.clear()
        pipeline.forward()
        return {
            "references_after_compile": counter_before,
            "cleanups": [e for e in events if e[0] == "cleanup"],
            "cleanup_count": pipeline.serving.cleanups,
            "active_after_forward": pipeline.active_llm_serving is None,
        }

    out["serving/lifecycle_refcount"] = _outcome(serving_lifecycle)

    def resume(step: int, tag: str):
        Pipe, events = build(f"resume_{tag}", nops=3)
        pipeline = Pipe()
        pipeline.compile()
        events.clear()
        pipeline.forward(resume_step=step)
        return [e[1] for e in events if e[0] == "run"]

    out["forward/resume_step_0"] = _outcome(lambda: resume(0, "zero"))
    out["forward/resume_step_1"] = _outcome(lambda: resume(1, "one"))
    return out


# --------------------------------------------------------------------------
# group 3: batched and streaming pipelines
# --------------------------------------------------------------------------

def batched_cases(api, root: Path) -> dict[str, Any]:
    """Probe batch slicing, append-on-later-batch writes and resume markers."""
    OperatorABC = api["OperatorABC"]
    out: dict[str, Any] = {}

    def run_batched(kind: str, batch_size, rows=6, resume_from_last=True, tag=""):
        PipelineCls = api["BatchedPipelineABC"] if kind == "batched" else api["StreamBatchedPipelineABC"]
        StorageCls = api["BatchedFileStorage"] if kind == "batched" else api["StreamBatchedFileStorage"]
        name = f"{kind}_{tag}"
        entry = _write(
            root / f"{name}.jsonl",
            "".join(json.dumps({"i": i}) + "\n" for i in range(rows)),
        )
        cache = root / name
        seen: list[Any] = []

        class Tagging(OperatorABC):
            def __init__(self):
                super().__init__()

            def run(self, storage):
                batch = storage.read("dict")
                seen.append([r["i"] for r in batch])
                storage.write([{**r, "seen": True} for r in batch])

        class Pipe(PipelineCls):
            def __init__(self):
                super().__init__()
                self.storage = StorageCls(
                    first_entry_file_name=str(entry),
                    cache_path=str(cache),
                    file_name_prefix="pipeline_step",
                    cache_type="jsonl",
                )
                self.tagging = Tagging()

            def forward(self):
                self.tagging.run(storage=self.storage.step())

        pipeline = Pipe()
        pipeline.compile()
        pipeline.forward(batch_size=batch_size, resume_from_last=resume_from_last)
        marker = cache / "pipeline_step_last_success_step.txt"
        return {
            "batches_seen": seen,
            "files": sorted(p.name for p in cache.glob("*")),
            "output": (cache / "pipeline_step_step1.jsonl").read_text(encoding="utf-8"),
            "marker": marker.read_text(encoding="utf-8") if marker.is_file() else None,
        }

    out["batched/size_2"] = _outcome(lambda: run_batched("batched", 2, tag="s2"))
    out["batched/size_4_uneven"] = _outcome(lambda: run_batched("batched", 4, rows=6, tag="s4"))
    out["batched/size_none"] = _outcome(lambda: run_batched("batched", None, tag="none"))
    out["batched/no_resume_marker"] = _outcome(
        lambda: run_batched("batched", 2, resume_from_last=False, tag="nomarker")
    )
    out["stream/size_2"] = _outcome(lambda: run_batched("stream", 2, tag="s2"))
    out["stream/size_4_uneven"] = _outcome(lambda: run_batched("stream", 4, rows=6, tag="s4"))

    # resume_step and resume_from_last are mutually exclusive
    def conflicting_resume():
        PipelineCls = api["BatchedPipelineABC"]
        StorageCls = api["BatchedFileStorage"]
        entry = _write(root / "conflict.jsonl", '{"i":0}\n')

        class Noop(OperatorABC):
            def __init__(self):
                super().__init__()

            def run(self, storage):
                storage.write(storage.read("dict"))

        class Pipe(PipelineCls):
            def __init__(self):
                super().__init__()
                self.storage = StorageCls(
                    first_entry_file_name=str(entry), cache_path=str(root / "conflict")
                )
                self.noop = Noop()

            def forward(self):
                self.noop.run(storage=self.storage.step())

        pipeline = Pipe()
        pipeline.compile()
        return pipeline.forward(resume_step=1, resume_from_last=True)

    out["batched/conflicting_resume"] = _outcome(conflicting_resume)

    out["batched/rejects_parquet_cache_type"] = _outcome(
        lambda: api["BatchedFileStorage"](
            first_entry_file_name=str(root / "conflict.jsonl"), cache_type="parquet"
        )
    )

    def stream_record_count():
        storage = api["StreamBatchedFileStorage"](
            first_entry_file_name=str(
                _write(root / "count.jsonl", '{"i":0}\n{"i":1}\n{"i":2}\n')
            )
        )
        view = storage.step()
        return {
            "count": view.get_record_count(),
            "cached": view.get_record_count(),
            "keys": view.get_keys_from_dataframe(),
            "chunks": [len(chunk) for chunk in view.iter_chunks()],
        }

    out["stream/record_count_and_keys"] = _outcome(stream_record_count)
    return out


# --------------------------------------------------------------------------
# group 4: serving request bodies and response formatting
# --------------------------------------------------------------------------

#: ``format_response`` inputs covering the tri-state policy and malformed bodies.
RESPONSE_FIXTURES: dict[str, Any] = {
    "reasoning_content": {"choices": [{"message": {"reasoning_content": "R", "content": "C"}}]},
    "reasoning_alias": {"choices": [{"message": {"reasoning": "R", "content": "C"}}]},
    "content_only": {"choices": [{"message": {"content": "C"}}]},
    "prewrapped": {
        "choices": [
            {"message": {"reasoning_content": "R", "content": "<think>x</think>\n<answer>y</answer>"}}
        ]
    },
    "null_content_with_reasoning": {
        "choices": [{"message": {"content": None, "reasoning_content": "R"}}]
    },
    "null_content_no_reasoning": {"choices": [{"message": {"content": None}}]},
    "empty_choices": {"choices": []},
    "no_choices_key": {},
    "message_not_mapping": {"choices": [{"message": "nope"}]},
    "empty_reasoning_string": {
        "choices": [{"message": {"reasoning_content": "", "content": "C"}}]
    },
}

#: SSE streams covering multi-choice ordering, aliases and UTF-8 decoding.
SSE_FIXTURES: dict[str, list[dict[str, Any]]] = {
    "single_choice": [
        {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "st "}}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning_content": "ep"}}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "ans"}}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 9}},
    ],
    "multi_choice_out_of_order": [
        {"choices": [{"index": 1, "delta": {"content": "B"}}, {"index": 0, "delta": {"content": "A"}}]},
        {"choices": [
            {"index": 0, "delta": {"content": "A2"}, "finish_reason": "stop"},
            {"index": 1, "delta": {"content": "B2"}, "finish_reason": "length"},
        ]},
    ],
    "reasoning_alias": [
        {"choices": [{"index": 0, "delta": {"reasoning": "RR", "content": "CC"}, "finish_reason": "stop"}]}
    ],
    "non_ascii": [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "2 × 3 → 6", "content": "\u2713"}, "finish_reason": "stop"}]}
    ],
    "metadata_fields": [
        {"id": "x", "model": "m", "created": 17, "system_fingerprint": "fp",
         "choices": [{"index": 0, "delta": {"content": "t"}, "finish_reason": "stop"}]}
    ],
}

ENABLE_THINKING_STATES = [
    ("absent", "__absent__"),
    ("none", None),
    ("omit", "omit"),
    ("true", True),
    ("false", False),
    ("invalid_string", "false"),
]


def serving_cases(api, root: Path) -> dict[str, Any]:
    """Probe constructor defaults, request bodies, retries and formatting."""
    make_serving = api["make_serving"]
    format_response = api["format_response"]
    aggregate_sse = api["aggregate_sse"]
    capture = api["capture_requests"]
    out: dict[str, Any] = {}

    # constructor defaults and missing-key behavior
    out["ctor/missing_key"] = _outcome(lambda: make_serving(key_present=False))

    def defaults():
        serving = make_serving()
        return {
            "configs": dict(serving.configs),
            "timeout": list(serving.timeout),
            "max_retries": serving.max_retries,
            "max_workers": serving.max_workers,
            "headers": dict(serving.headers),
            "has_session": serving.session is not None,
        }

    out["ctor/defaults"] = _outcome(defaults)
    out["ctor/deprecated_timeout"] = _outcome(
        lambda: list(make_serving(timeout=33).timeout)
    )
    out["ctor/extra_configs"] = _outcome(
        lambda: dict(make_serving(stream=True, top_p=0.9, temperature=0.7).configs)
    )
    for label, value in ENABLE_THINKING_STATES:
        if value == "__absent__":
            out[f"ctor/enable_thinking_{label}"] = _outcome(
                lambda: dict(make_serving().configs)
            )
        else:
            out[f"ctor/enable_thinking_{label}"] = _outcome(
                lambda v=value: dict(make_serving(enable_thinking=v).configs)
            )

    # request bodies
    def body(**kwargs):
        method = kwargs.pop("_method", "generate_from_input")
        args = kwargs.pop("_args", (["hello"],))
        call_kwargs = kwargs.pop("_kwargs", {})
        with capture() as recorded:
            serving = make_serving(**kwargs)
            result = getattr(serving, method)(*args, **call_kwargs)
        return {"request": recorded[0], "result": result}

    out["request/default"] = _outcome(body)
    out["request/enable_thinking_true"] = _outcome(lambda: body(enable_thinking=True))
    out["request/enable_thinking_false"] = _outcome(lambda: body(enable_thinking=False))
    out["request/enable_thinking_omit"] = _outcome(lambda: body(enable_thinking="omit"))
    out["request/custom_system_prompt"] = _outcome(
        lambda: body(_kwargs={"system_prompt": "SYS"})
    )
    out["request/json_schema"] = _outcome(
        lambda: body(_kwargs={"json_schema": {"type": "object"}})
    )
    out["request/conversations"] = _outcome(
        lambda: body(
            _method="generate_from_conversations",
            _args=([[{"role": "user", "content": "q"}]],),
        )
    )
    out["request/embedding"] = _outcome(
        lambda: body(_method="generate_embedding_from_input", _args=(["t"],))
    )
    out["request/multiple_prompts"] = _outcome(
        lambda: body(_args=(["one", "two", "three"],), max_workers=3)
    )

    # response formatting across the tri-state policy
    for label, state in ENABLE_THINKING_STATES:
        for name, payload in RESPONSE_FIXTURES.items():
            out[f"format/{label}/{name}"] = _outcome(
                lambda p=payload, s=state: format_response(p, s)
            )

    # SSE aggregation, then the formatted text taken from it
    for name, chunks in SSE_FIXTURES.items():
        out[f"sse/{name}"] = _outcome(lambda c=chunks: aggregate_sse(c))
        out[f"sse/{name}/formatted_true"] = _outcome(
            lambda c=chunks: format_response(aggregate_sse(c), True)
        )

    # failure classification and retry behavior
    out["failure/http_500_retries"] = _outcome(
        lambda: api["failure_probe"]("http_500", max_retries=3)
    )
    out["failure/read_timeout"] = _outcome(
        lambda: api["failure_probe"]("read_timeout", max_retries=1)
    )
    out["failure/connect_timeout"] = _outcome(
        lambda: api["failure_probe"]("connect_timeout", max_retries=1)
    )
    out["failure/connection_error"] = _outcome(
        lambda: api["failure_probe"]("connection_error", max_retries=1)
    )
    out["failure/connection_error_read_timed_out"] = _outcome(
        lambda: api["failure_probe"]("connection_error_read_timed_out", max_retries=1)
    )
    out["failure/malformed_json"] = _outcome(
        lambda: api["failure_probe"]("malformed_json", max_retries=1)
    )

    # results stay aligned with inputs regardless of completion order
    out["concurrency/result_order"] = _outcome(lambda: api["ordering_probe"]())
    return out


CASE_GROUPS: dict[str, Callable[[Any, Path], dict[str, Any]]] = {
    "storage": storage_cases,
    "pipeline": pipeline_cases,
    "batched": batched_cases,
    "serving": serving_cases,
}

#: Groups whose results depend on pandas type inference, so a pandas major
#: version change can legitimately move them.
PANDAS_SENSITIVE_GROUPS = frozenset({"storage", "batched"})


def evaluate(api, root: Path, groups: list[str] | None = None) -> dict[str, Any]:
    """Evaluate the selected case groups against one implementation."""
    selected = groups or list(CASE_GROUPS)
    results: dict[str, Any] = {}
    for name in selected:
        group_root = Path(root) / name
        group_root.mkdir(parents=True, exist_ok=True)
        results[name] = CASE_GROUPS[name](api, group_root)
    return results
