"""Compile/forward pipeline aligned with ``open-dataflow`` 1.0.10 ``PipelineABC``.

The execution contract generated code is written against:

* ``compile()`` replaces every :class:`OperatorABC` member with a recording
  proxy, then calls ``forward()`` **once**. Operator bodies do not run during
  that call, but everything else in ``forward()`` does — including the
  ``storage.step()`` calls passed as arguments, which is what fixes each
  operator's step number at compile time. No cache file is written.
* After ``compile()``, ``forward`` is rebound to ``_compiled_forward``, which
  executes the recorded operators in order.
* Key integrity is validated at compile time from ``input_*``/``output_*``
  string arguments, accumulated across the graph starting from the first
  storage's columns.

Every behavior here was measured against the real installed package; the
measurements are pinned in ``tests/parity``. Keep non-operator side effects out
of ``forward()``: they run twice, once per phase.

Not reproduced: ``draw_graph``, the reference's web visualization entry point.
It starts an HTTP server and opens a browser, which conflicts with headless
automated runs, and it is presentation rather than execution semantics. The
graph structures it renders (``op_nodes_list``, ``accumulated_keys``,
``final_keys``, ``last_modified_index_of_keys``) are all built and available.
"""

from __future__ import annotations

import copy
import inspect
import os
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Any, Dict

from tqdm import tqdm

from .llm_serving import LLMServingABC
from .dataflow_operator import OperatorABC
from .runtime_logger import get_logger
from ..io.dataflow_storage import DataFlowStorage


class KeyNode:
    """One key at one position in the graph, linked to where it was last set."""

    def __init__(self, key_para_name: str, key: str, ptr: list["KeyNode"] | None = None):
        self.key_para_name = key_para_name
        self.key = key
        self.ptr: list[KeyNode] = ptr if ptr is not None else []

    def set_index(self, index: int) -> None:
        self.index = index

    def __str__(self) -> str:
        ptr_status = (
            [(node.key, getattr(node, "index", None), hex(id(node))) for node in self.ptr]
            if self.ptr
            else ["None"]
        )
        ptr_str = "".join(f"\n      <{item}>" for item in ptr_status)
        return (
            f"\n    KeyNode[{hex(id(self))}](key_para_name={self.key_para_name}, "
            f"key={self.key}, ptr_keys={ptr_str})"
        )

    __repr__ = __str__


class OperatorNode:
    """One recorded operator call plus the keys it reads and writes."""

    def __init__(
        self,
        op_obj: OperatorABC | None = None,
        op_name: str | None = None,
        storage: DataFlowStorage | None = None,
        llm_serving: LLMServingABC | None = None,
        **kwargs: Any,
    ) -> None:
        self.op_obj = op_obj
        self.op_name = op_name
        self.storage = storage
        self.llm_serving = llm_serving
        self.kwargs = kwargs

        self.input_keys: list[str] = []
        self.input_key_nodes: dict[str, KeyNode] = {}
        self.output_keys: list[str] = []
        self.output_keys_nodes: dict[str, KeyNode] = {}

        self._get_keys_from_kwargs()

    def _get_keys_from_kwargs(self) -> None:
        for key, value in self.kwargs.items():
            if key.startswith("input_") and isinstance(value, str):
                self.input_keys.append(value)
                self.input_key_nodes[value] = KeyNode(key, value)
            elif key.startswith("output_") and isinstance(value, str):
                self.output_keys.append(value)
                self.output_keys_nodes[value] = KeyNode(key, value)
            else:
                # Not a declared key; it is still forwarded to run().
                print(
                    f"\033[91mWarning: Unexpected key '{key}' in operator "
                    f"{self.op_obj.__class__.__name__}\033[0m"
                )

    def init_output_keys_nodes(self, keys: list[str]) -> None:
        for key in keys:
            self.output_keys.append(key)
            self.output_keys_nodes[key] = KeyNode(key, key)

    def init_input_keys_nodes(self, keys: list[str]) -> None:
        for key in keys:
            self.input_keys.append(key)
            self.input_key_nodes[key] = KeyNode(key, key)

    def __str__(self) -> str:
        op_class = self.op_obj.__class__.__name__ if self.op_obj else None
        return (
            f"OperatorNode(\n"
            f"  op_obj:{self.op_obj}"
            f"  Operator_class: {op_class},\n"
            f"  Operator_name: {self.op_name},\n"
            f"  Storage: {self.storage},\n"
            f"  Input Keys: [{', '.join(self.input_keys)}],\n"
            f"  Output Keys: [{', '.join(self.output_keys)}],\n"
            f"  Input Nodes: [{self.input_key_nodes}],\n"
            f"  Output Nodes: [{self.output_keys_nodes}],\n"
            f"  Additional Params: {self.kwargs}\n"
            f")"
        )


class OPRuntime:
    """One ``operator.run(...)`` call captured during compile."""

    def __init__(self, operator: OperatorABC, operator_name: str, args: Dict[str, Any]):
        self.op = operator
        self.op_name = operator_name
        self.kwargs = args
        self.logger = get_logger()

    def __repr__(self) -> str:
        return f"OPRuntime(operator={repr(self.op)}, args={self.kwargs})"


class AutoOP:
    """Compile-time stand-in that records a call instead of running it.

    Binds against the real ``run`` signature so a bad call fails at compile
    time with the same ``TypeError`` the reference raises, and flattens
    ``**kwargs`` to top level so key arguments are visible to the graph builder.
    """

    def __init__(self, operator: OperatorABC, operator_name: str, pipeline: "PipelineABC"):
        self._operator = operator
        self._operator_name = operator_name
        self._pipeline = pipeline
        self._logger = get_logger()

        self._orig_run = operator.run
        self._signature = inspect.signature(operator.run)
        self.__doc__ = self._orig_run.__doc__

    def _flatten_bound_arguments(
        self, bound: inspect.BoundArguments, sig: inspect.Signature
    ) -> "OrderedDict[str, object]":
        """Flatten ``**kwargs`` into the top level, preserving call order."""
        var_kw_name = None
        for name, parameter in sig.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw_name = name
                break

        out: "OrderedDict[str, object]" = OrderedDict()
        for name, value in bound.arguments.items():
            if name == var_kw_name and isinstance(value, dict):
                for key, item in value.items():
                    if key in out:
                        out[f"__kw__{key}"] = item
                    else:
                        out[key] = item
            else:
                out[name] = value
        return out

    def run(self, *args: Any, **kwargs: Any) -> None:
        self._signature = inspect.signature(self._orig_run)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        final_kwargs = self._flatten_bound_arguments(bound_args, self._signature)
        self._logger.debug(final_kwargs)
        self._pipeline.op_runtimes.append(
            OPRuntime(
                operator=self._operator,
                operator_name=self._operator_name,
                args=dict(final_kwargs),
            )
        )


class PipelineABC(ABC):
    """Base class for a generated pipeline."""

    def __init__(self) -> None:
        self.op_runtimes: list[OPRuntime] = []
        self.compiled = False
        # accumulated_keys[0] holds the keys available before the first
        # operator; element i+1 holds them after operator i.
        self.accumulated_keys: list[list[str]] = []

        self.logger = get_logger()
        self.active_llm_serving = None

        self.op_nodes_list: list[OperatorNode] = []
        self.llm_serving_list: list[LLMServingABC | None] = []
        self.llm_serving_counter: Counter = Counter()

    @abstractmethod
    def forward(self):
        """Ordered operator calls. Keep other side effects out: this runs twice."""

    def compile(self):
        """Record the operator graph, then validate key integrity.

        Calling this twice fails: the first pass popped ``storage`` out of every
        recorded call, so the second finds ``None`` where a storage must be. The
        assertion is the reference's and is reproduced rather than replaced with
        a friendlier error, because generated code may depend on the failure
        type. Construct a new pipeline instead of recompiling one.
        """
        self.compiled = True
        for name, value in vars(self).items():
            if isinstance(value, OperatorABC):
                setattr(self, name, AutoOP(value, name, self))
        self.forward()
        # AutoOP callbacks have now appended one OPRuntime per operator call.

        self.forward = self._compiled_forward
        self.logger.info(
            f"Compiling pipeline and validating key integrity "
            f"across {len(self.op_runtimes)} operator runtimes."
        )
        self._build_operator_nodes_graph()

    def _build_operator_nodes_graph(self) -> None:
        """Turn recorded calls into nodes and validate key integrity."""
        for op_runtime in self.op_runtimes:
            llm_serving_obj, storage_obj = None, None
            for _, value in vars(op_runtime.op).items():
                if isinstance(value, LLMServingABC):
                    llm_serving_obj = value
            storage_obj = op_runtime.kwargs.pop("storage", None)

            assert isinstance(storage_obj, DataFlowStorage), (
                f"Storage must be a DataFlowStorage object, but got {type(storage_obj)} "
                f"in {op_runtime}'s `run` function with key `storage`."
            )

            op_node = OperatorNode(
                op_obj=op_runtime.op,
                op_name=op_runtime.op_name,
                storage=storage_obj,
                llm_serving=llm_serving_obj,
                **op_runtime.kwargs,
            )

            self.op_nodes_list.append(op_node)
            self.llm_serving_list.append(llm_serving_obj)
            if llm_serving_obj is not None:
                self.llm_serving_counter[llm_serving_obj] += 1
        self.logger.debug(
            f"Built operator nodes graph with {self.op_nodes_list} nodes, \n"
            f"and {self.llm_serving_list} LLM Serving objects."
        )

        first_op = self.op_nodes_list[0] if self.op_nodes_list else None
        if first_op and first_op.storage:
            iter_storage_keys = first_op.storage.get_keys_from_dataframe()
        else:
            iter_storage_keys = []

        self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))

        error_msg = []
        for op_node in self.op_nodes_list:
            for input_key in op_node.input_keys:
                if input_key not in self.accumulated_keys[-1]:
                    error_msg.append(
                        {
                            "input_key": input_key,
                            "op_name": op_node.op_name,
                            "class_name": op_node.op_obj.__class__.__name__,
                            "key_para_name": op_node.input_key_nodes[
                                input_key
                            ].key_para_name,
                        }
                    )

            for output_key in op_node.output_keys:
                if output_key not in iter_storage_keys:
                    iter_storage_keys.append(output_key)
            self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))

        if len(error_msg) != 0:
            details = "\n".join(
                f"- Input key '{e['input_key']}' in `{e['op_name']}` "
                f"(class <{e['class_name']}>) does not match any output keys "
                f"from previous operators or dataset keys. "
                f"Check parameter '{e['key_para_name']}' in the `{e['op_name']}.run()`."
                for e in error_msg
            )
            msg = (
                "Key Matching Error in following Operators during pipeline.compile():\n"
                f"{details}"
            )
            self.logger.warning(msg)
            raise KeyError(msg)

        self.final_keys = copy.deepcopy(iter_storage_keys)
        self.logger.debug(f"Accumulated keys after building graph: {self.accumulated_keys}")

        self.input_dataset_node = OperatorNode(None, "DATASET-INPUT", None, None)
        self.input_dataset_node.init_output_keys_nodes(self.accumulated_keys[0])
        self.op_nodes_list.insert(0, self.input_dataset_node)

        self.output_dataset_node = OperatorNode(None, "DATASET-OUTPUT", None, None)
        self.output_dataset_node.init_input_keys_nodes(self.final_keys)
        self.op_nodes_list.append(self.output_dataset_node)

        self.last_modified_index_of_keys: dict[str, list[int]] = {
            key: [] for key in self.final_keys
        }

        # Index 0 is now the DATASET-INPUT node.
        for idx, i_op in enumerate(self.op_nodes_list):
            for input_key in i_op.input_keys:
                current_keynode = i_op.input_key_nodes[input_key]
                current_keynode.set_index(idx)

                if len(self.last_modified_index_of_keys[input_key]) > 0:
                    last_modified_idx = self.last_modified_index_of_keys[input_key][-1]
                    last_modified_keynode = self.op_nodes_list[
                        last_modified_idx
                    ].output_keys_nodes[input_key]
                    # Both directions, so the graph can be walked either way.
                    last_modified_keynode.ptr.append(current_keynode)
                    current_keynode.ptr.append(last_modified_keynode)
            for output_key in i_op.output_keys:
                current_keynode = i_op.output_keys_nodes[output_key]
                current_keynode.set_index(idx)
                self.last_modified_index_of_keys[output_key].append(idx)

        for op in self.op_nodes_list:
            self.logger.debug(f"Operator Node: {op}")

    def _activate_serving(self, op_node: OperatorNode) -> None:
        """Make this node's serving active, cleaning up a different one first."""
        if self.active_llm_serving and self.active_llm_serving is not op_node.llm_serving:
            self.logger.debug(
                f"Detected active LLM Serving {self.active_llm_serving}, "
                f"new serving {op_node.llm_serving}, cleaning up..."
            )
            self.active_llm_serving.cleanup()
        self.active_llm_serving = op_node.llm_serving

    def _release_serving(self) -> None:
        """Drop one reference and clean up when the last one is released."""
        self.llm_serving_counter[self.active_llm_serving] -= 1
        if self.llm_serving_counter[self.active_llm_serving] == 0:
            self.logger.debug(
                f"Detected LLM Serving {self.active_llm_serving} ref reduced to 0, "
                "cleaning up..."
            )
            self.active_llm_serving.cleanup()
            self.active_llm_serving = None

    def _compiled_forward(self, resume_step: int = 0):
        """Run the recorded operators in order.

        ``resume_step`` skips the first N operators; the ``idx - 1`` offset is
        the DATASET-INPUT node inserted at position 0.
        """
        for idx, op_node in enumerate(self.op_nodes_list):
            if idx - 1 < resume_step:
                continue

            self.logger.debug(
                f"Ready to run {op_node}, with serving={op_node.llm_serving}, "
                f"active_llm_serving={self.active_llm_serving}"
            )
            if op_node.llm_serving is not None:
                self._activate_serving(op_node)

            if op_node.op_obj is not None:
                op_node.op_obj.run(storage=op_node.storage, **op_node.kwargs)

            if op_node.llm_serving is not None:
                self._release_serving()


class BatchedPipelineABC(PipelineABC):
    """Pipeline that runs each operator over fixed-size batches.

    With ``resume_from_last``, progress is recorded in
    ``{cache_path}/{file_name_prefix}_last_success_step.txt`` as
    ``"{step},{batch}"`` after every batch and again after every operator, so an
    interrupted run resumes mid-operator. Requires a
    :class:`~rsi.framework.io.dataflow_storage.BatchedFileStorage`.
    """

    def __init__(self) -> None:
        super().__init__()

    def _resume_marker_path(self) -> str:
        storage = self.op_nodes_list[1].storage
        return os.path.join(
            storage.cache_path, f"{storage.file_name_prefix}_last_success_step.txt"
        )

    def _resolve_resume(
        self, resume_step: int, resume_from_last: bool
    ) -> tuple[int, int, str | None]:
        if not self.compiled:
            raise RuntimeError(
                "Pipeline is not compiled yet. Please call `compile()` before "
                "running the pipeline."
            )
        if resume_step > 0 and resume_from_last:
            raise ValueError(
                "Cannot set both `resume_step` and `resume_from_last` to True."
            )
        resume_batch = 0
        cache_path = None
        if resume_from_last:
            cache_path = self._resume_marker_path()
            if not os.path.exists(cache_path):
                resume_step, resume_batch = 0, 0
                self.logger.info(
                    f"No last success step cache found at {cache_path}, "
                    "starting from step 0."
                )
            else:
                with open(cache_path, "r") as handle:
                    line = handle.readline().strip()
                    resume_step, resume_batch = map(int, line.split(","))
                self.logger.info(
                    f"Resuming from last success step {resume_step}, "
                    f"batch step {resume_batch}."
                )
        return resume_step, resume_batch, cache_path

    @staticmethod
    def _write_marker(cache_path: str, step: int, batch: int) -> None:
        with open(cache_path, "w") as handle:
            handle.write(f"{step},{batch}\n")

    def _compiled_forward(
        self,
        resume_step: int = 0,
        batch_size: int | None = None,
        resume_from_last: bool = True,
    ):
        resume_step, resume_batch, cache_path = self._resolve_resume(
            resume_step, resume_from_last
        )

        for idx, op_node in enumerate(self.op_nodes_list):
            if idx - 1 < resume_step:
                continue

            self.logger.debug(
                f"Ready to run {op_node}, with serving={op_node.llm_serving}, "
                f"active_llm_serving={self.active_llm_serving}"
            )
            if op_node.llm_serving is not None:
                self._activate_serving(op_node)

            if op_node.op_obj is not None:
                record_count = 0
                if batch_size is not None:
                    storage = op_node.storage
                    storage.batch_step = 0 if idx - 1 > resume_step else resume_batch
                    storage.batch_size = batch_size
                    storage.read()  # read to set record_count
                    record_count = storage.record_count

                run_times = (
                    1
                    if batch_size is None
                    else ((record_count - 1) // batch_size + 1) - op_node.storage.batch_step
                )
                if batch_size is not None:
                    self.logger.info(
                        f"Pipeline will run for {run_times} iterations to cover "
                        f"{record_count} records with batch size {batch_size}."
                    )
                for _ in tqdm(
                    range(run_times),
                    desc=f"\033[1;36mRunning {op_node.op_name} with batch size={batch_size}\033[0m",
                    position=0,
                    dynamic_ncols=True,
                    colour="cyan",
                ):
                    op_node.op_obj.run(storage=op_node.storage, **op_node.kwargs)
                    if batch_size is not None:
                        op_node.storage.batch_step += 1
                    if resume_from_last:
                        resume_batch = (
                            op_node.storage.batch_step if batch_size is not None else 0
                        )
                        self._write_marker(cache_path, idx - 1, resume_batch)
            if resume_from_last:
                resume_batch = 0  # reset for the next operator
                self._write_marker(cache_path, idx, resume_batch)
            if op_node.llm_serving is not None:
                self._release_serving()


class StreamBatchedPipelineABC(BatchedPipelineABC):
    """Batched pipeline that streams each step instead of loading it whole.

    Chunks come from
    :meth:`~rsi.framework.io.dataflow_storage.StreamBatchedFileStorage.iter_chunks`
    and are handed to the operator through ``_current_streaming_chunk``, so a
    step larger than memory can still be processed.
    """

    def __init__(self) -> None:
        super().__init__()

    def _compiled_forward(
        self,
        resume_step: int = 0,
        batch_size: int | None = None,
        resume_from_last: bool = True,
    ):
        resume_step, resume_batch, cache_path = self._resolve_resume(
            resume_step, resume_from_last
        )

        for idx, op_node in enumerate(self.op_nodes_list):
            if idx - 1 < resume_step:
                continue

            self.logger.debug(
                f"Ready to run {op_node}, with serving={op_node.llm_serving}, "
                f"active_llm_serving={self.active_llm_serving}"
            )
            if op_node.llm_serving is not None:
                self._activate_serving(op_node)

            if op_node.op_obj is not None:
                record_count = 0
                data_stream = None
                if batch_size is not None:
                    storage = op_node.storage
                    storage.batch_step = 0 if idx - 1 > resume_step else resume_batch
                    storage.batch_size = batch_size
                    record_count = storage.get_record_count()
                    data_stream = storage.iter_chunks()
                    for _ in range(storage.batch_step):
                        next(data_stream, None)

                run_times = (
                    1
                    if batch_size is None
                    else ((record_count - 1) // batch_size + 1) - op_node.storage.batch_step
                )
                if batch_size is not None:
                    self.logger.info(
                        f"Pipeline will run for {run_times} iterations to cover "
                        f"{record_count} records with batch size {batch_size}."
                    )
                for _ in tqdm(
                    range(run_times),
                    desc=f"\033[1;36mRunning {op_node.op_name} with batch size={batch_size}\033[0m",
                    position=0,
                    dynamic_ncols=True,
                    colour="cyan",
                ):
                    if batch_size is not None:
                        try:
                            op_node.storage._current_streaming_chunk = next(data_stream)
                        except StopIteration:
                            break

                    op_node.op_obj.run(storage=op_node.storage, **op_node.kwargs)

                    if batch_size is not None:
                        op_node.storage._current_streaming_chunk = None
                        op_node.storage.batch_step += 1
                    if resume_from_last:
                        resume_batch = (
                            op_node.storage.batch_step if batch_size is not None else 0
                        )
                        self._write_marker(cache_path, idx - 1, resume_batch)
            if resume_from_last:
                resume_batch = 0  # reset for the next operator
                self._write_marker(cache_path, idx, resume_batch)
            if op_node.llm_serving is not None:
                self._release_serving()
