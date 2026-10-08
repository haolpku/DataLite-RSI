"""Step-based storage aligned with ``open-dataflow`` 1.0.10 ``FileStorage``.

Generated operators and pipelines are written against this contract, so its
observable behavior must match the reference runtime rather than redefine it.
Every rule below was measured against the real installed package in an isolated
environment; ``tests/parity`` pins those measurements.

Type semantics come from pandas inference, which is why pandas is a dependency:
reimplementing the coercion rules (``int`` column with a null becoming
``float``, the numeric string ``"12"`` becoming ``12``, a ``date`` column
becoming ``Timestamp``) would be a different runtime wearing the same name.

The entry is always a local file. The reference also accepted ``hf:``/``ms:``
dataset identifiers; the evolution loop forbids fetching data over the network
and operates on a fixed local corpus, so that path does not exist here and such
a value is just a missing file.

One deliberate difference from the reference, documented in
``docs/dataflow-serving-parity.md``: ``write()`` before ``step()`` raises. The
reference resolves step ``-1 + 1`` to the ``first_entry_file_name`` and would
overwrite the fixed corpus in place, which breaks the immutable-input premise
of the loop.

The upstream ``compat/storage.py`` fix is built in rather than patched:
``read("dict")`` restores pandas missing values to Python ``None``, because JSON
has a single missing-value representation and ``NaN``/``NaT``/``pd.NA`` are
unsafe in ordinary operator fallback expressions such as
``row.get("a") or row["b"]``.
"""

from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Generator, Literal

import pandas as pd

from ..core.runtime_logger import get_logger


def clean_surrogates(obj: Any) -> Any:
    """Replace unpaired surrogates so a row can be serialized as UTF-8.

    Mirrors the reference ``write()`` helper, including its fallback of
    stringifying values that are neither containers, numbers, bools nor None.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, dict):
        return {key: clean_surrogates(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_surrogates(item) for item in obj]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    try:
        return clean_surrogates(str(obj))
    except Exception:
        return obj


def dataframe_records_with_python_nulls(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to records without leaking pandas missing values."""
    normalized = dataframe.astype(object).where(dataframe.notna(), None)
    return normalized.to_dict(orient="records")


def _frame_from_rows(data: Any) -> pd.DataFrame:
    """Build the write frame, reproducing the reference's input validation."""
    if isinstance(data, list):
        # An empty list indexes [0] in the reference and raises IndexError.
        if len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame([clean_surrogates(item) for item in data])
        raise ValueError(f"Unsupported data type: {type(data[0])}")
    if isinstance(data, pd.DataFrame):
        return data.map(clean_surrogates)
    raise ValueError(f"Unsupported data type: {type(data)}")


class DataFlowStorage(ABC):
    """Abstract base class for data storage."""

    @abstractmethod
    def get_keys_from_dataframe(self) -> list[str]:
        """Return the column names of the data held at the current step."""

    @abstractmethod
    def read(self, output_type) -> Any:
        """Read the current step as the requested type."""

    @abstractmethod
    def write(self, data: Any) -> Any:
        """Write the next step and return its path."""

    def __repr__(self) -> str:
        parts = []
        for key, value in self.__dict__.items():
            if isinstance(value, pd.DataFrame):
                shown = f"<DataFrame shape={value.shape}>"
            elif isinstance(value, (set, dict)):
                shown = f"<{type(value).__name__} size={len(value)}>"
            else:
                shown = repr(value)
                if len(shown) > 100:
                    shown = shown[:97] + "..."
            parts.append(f"  {key} = {shown}")
        return f"{type(self).__name__}(\n" + "\n".join(parts) + "\n)"


class FileStorage(DataFlowStorage):
    """Disk-backed, step-based storage.

    Step 0 reads ``first_entry_file_name`` and its format is taken from that
    file's extension. Steps >= 1 live under ``cache_path`` as
    ``{file_name_prefix}_step{N}.{cache_type}``. ``operator_step`` starts at
    ``-1``; ``step()`` advances it and returns a shallow copy, so the view
    handed to one operator keeps that operator's step number even as the parent
    advances. Nothing is buffered between steps.
    """

    def __init__(
        self,
        first_entry_file_name: str,
        cache_path: str = "./cache",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: Literal["json", "jsonl", "csv", "parquet", "pickle"] = "jsonl",
    ) -> None:
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1
        self.logger = get_logger()

    def _get_cache_file_path(self, step) -> str:
        if step == -1:
            message = (
                "You must call storage.step() before reading or writing data. "
                "Please call storage.step() first for each operator step."
            )
            self.logger.error(message)
            raise ValueError(message)
        if step == 0:
            return os.path.join(self.first_entry_file_name)
        return os.path.join(
            self.cache_path, f"{self.file_name_prefix}_step{step}.{self.cache_type}"
        )

    def step(self):
        self.operator_step += 1
        return copy.copy(self)

    def reset(self):
        self.operator_step = -1
        return self

    def get_keys_from_dataframe(self) -> list[str]:
        dataframe = self.read(output_type="dataframe")
        return dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []

    def _load_local_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load a local file by type, wrapping every failure as ValueError."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} does not exist. Please check the path."
            )
        try:
            if file_type == "json":
                return pd.read_json(file_path)
            elif file_type == "jsonl":
                return pd.read_json(file_path, lines=True)
            elif file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "pickle":
                return pd.read_pickle(file_path)
            # Reproduces the reference's quirk: this branch tests cache_type,
            # not the file_type argument.
            elif self.cache_type == "xlsx":
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_type} file: {str(e)}")

    def _convert_output(self, dataframe: pd.DataFrame, output_type: str) -> Any:
        """Convert the frame to the requested output type.

        ``dict`` restores pandas missing values to Python ``None``; ``dataframe``
        deliberately keeps pandas missing-value semantics, where they are
        expected.
        """
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe_records_with_python_nulls(dataframe)
        raise ValueError(f"Unsupported output type: {output_type}")

    def _resolve_read_format(self, file_path: str) -> str:
        """Resolve the read format: the file extension at step 0, else cache_type."""
        if self.operator_step != 0:
            return self.cache_type
        return file_path.split(".")[-1]

    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Any:
        if self.operator_step == 0 and self.first_entry_file_name == "":
            self.logger.info("first_entry_file_name is empty, returning empty dataframe")
            return self._convert_output(pd.DataFrame(), output_type)

        file_path = self._get_cache_file_path(self.operator_step)
        self.logger.info(f"Reading data from {file_path} with type {output_type}")
        local_cache = self._resolve_read_format(file_path)
        dataframe = self._load_local_file(file_path, local_cache)
        return self._convert_output(dataframe, output_type)

    def write(self, data: Any) -> Any:
        """Write the next step immediately and return its path."""
        if self.operator_step == -1:
            # The reference would resolve step 0 here and overwrite the fixed
            # corpus in place. The loop treats that input as immutable.
            raise ValueError(
                "You must call storage.step() before reading or writing data. "
                "Writing before the first step would overwrite "
                f"first_entry_file_name ({self.first_entry_file_name!r})."
            )
        dataframe = _frame_from_rows(data)
        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger.success(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "json":
            dataframe.to_json(file_path, orient="records", force_ascii=False, indent=2)
        elif self.cache_type == "jsonl":
            dataframe.to_json(file_path, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            dataframe.to_csv(file_path, index=False)
        elif self.cache_type == "parquet":
            dataframe.to_parquet(file_path)
        elif self.cache_type == "pickle":
            dataframe.to_pickle(file_path)
        elif self.cache_type == "xlsx":
            dataframe.to_excel(file_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file type: {self.cache_type}, output file should end "
                "with json, jsonl, csv, parquet, pickle or xlsx"
            )
        return file_path


class BatchedFileStorage(FileStorage):
    """Step storage that reads and writes one batch at a time.

    Used by :class:`~rsi.framework.core.pipeline.BatchedPipelineABC`: the
    pipeline sets ``batch_size``/``batch_step``, each ``read`` returns that
    slice with a fresh index, and each ``write`` appends unless ``batch_step``
    is 0. The full frame per step is buffered so repeated reads of one step do
    not re-parse the file.
    """

    def __init__(
        self,
        first_entry_file_name: str,
        cache_path: str = "./cache",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: Literal["jsonl", "csv"] = "jsonl",
    ) -> None:
        super().__init__(first_entry_file_name, cache_path, file_name_prefix, cache_type)
        self.batch_size = None
        self.batch_step = 0
        self._dataframe_buffer: dict[int, pd.DataFrame] = {}
        if cache_type not in ["jsonl", "csv"]:
            raise ValueError(
                "BatchedFileStorage only supports 'jsonl' and 'csv' cache types, "
                f"got: {cache_type}"
            )

    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Any:
        if self.operator_step == 0 and self.first_entry_file_name == "":
            self.logger.info("first_entry_file_name is empty, returning empty dataframe")
            return self._convert_output(pd.DataFrame(), output_type)

        file_path = self._get_cache_file_path(self.operator_step)
        self.logger.info(f"Reading data from {file_path} with type {output_type}")
        local_cache = self._resolve_read_format(file_path)

        if self._dataframe_buffer.get(self.operator_step) is not None:
            dataframe = self._dataframe_buffer[self.operator_step].copy()
        else:
            dataframe = self._load_local_file(file_path, local_cache)
            self._dataframe_buffer[self.operator_step] = dataframe.copy()
        self.record_count = len(dataframe)
        if self.batch_size:
            # reset_index so each batch is indexed from 0 rather than carrying
            # its offset in the full frame.
            dataframe = dataframe.iloc[
                self.batch_step * self.batch_size : (self.batch_step + 1) * self.batch_size
            ].reset_index(drop=True)
        return self._convert_output(dataframe, output_type)

    def write(self, data: Any) -> Any:
        if self.operator_step == -1:
            raise ValueError(
                "You must call storage.step() before reading or writing data. "
                "Writing before the first step would overwrite "
                f"first_entry_file_name ({self.first_entry_file_name!r})."
            )
        dataframe = _frame_from_rows(data)
        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger.success(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "jsonl":
            open_mode = "w" if self.batch_step == 0 else "a"
            with open(file_path, open_mode, encoding="utf-8") as handle:
                dataframe.to_json(handle, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            if self.batch_step == 0:
                dataframe.to_csv(file_path, index=False)
            else:
                dataframe.to_csv(file_path, index=False, header=False, mode="a")
        else:
            raise ValueError(
                f"Unsupported file type: {self.cache_type}, output file should end "
                "with jsonl, csv"
            )
        return file_path


class StreamBatchedFileStorage(BatchedFileStorage):
    """Batched storage that streams chunks instead of loading the whole step.

    ``iter_chunks`` yields the step as a chunk generator and the pipeline hands
    each chunk back through ``_current_streaming_chunk``, so a step larger than
    memory can still be processed one batch at a time.
    """

    def __init__(
        self,
        first_entry_file_name: str,
        cache_path: str = "./cache",
        file_name_prefix: str = "dataflow_cache_step",
        cache_type: Literal["jsonl", "csv"] = "jsonl",
    ) -> None:
        super().__init__(first_entry_file_name, cache_path, file_name_prefix, cache_type)

    def get_keys_from_dataframe(self) -> list[str]:
        """Read only the header, falling back to an empty list on failure."""
        if self._dataframe_buffer.get(self.operator_step) is not None:
            return self._dataframe_buffer[self.operator_step].columns.tolist()

        file_path = self._get_cache_file_path(self.operator_step)
        ext = (
            self.cache_type
            if self.operator_step != 0
            else file_path.split(".")[-1].lower()
        )
        try:
            if ext == "csv":
                return pd.read_csv(file_path, nrows=0).columns.tolist()
            elif ext == "jsonl":
                with open(file_path, "r", encoding="utf-8") as handle:
                    line = handle.readline()
                    if line:
                        return list(json.loads(line).keys())
                return []
            elif ext == "parquet":
                # Parquet keeps its schema in the footer, so this is near-free.
                import pyarrow.parquet as pq

                return pq.ParquetFile(file_path).schema.names
        except Exception as e:
            self.logger.warning(
                f"Failed to read header from {file_path} directly: {e}. "
                "Falling back to read()."
            )
        return []

    def _load_local_file(
        self, file_path: str, file_type: str, batchsize: int = None
    ) -> pd.DataFrame:
        """Load a file, returning a chunk reader when a batch size is given."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {file_path} does not exist. Please check the path."
            )
        try:
            if file_type == "jsonl":
                return pd.read_json(file_path, lines=True, chunksize=batchsize)
            elif file_type == "csv":
                return pd.read_csv(file_path, chunksize=batchsize)
            return super()._load_local_file(file_path, file_type)
        except Exception as e:
            raise ValueError(f"Failed to load {file_type} file: {str(e)}")

    def get_record_count(self) -> int:
        """Count records without materializing the step, cached per instance."""
        if hasattr(self, "_cached_record_count"):
            return self._cached_record_count

        file_path = self._get_cache_file_path(self.operator_step)
        local_cache = (
            self.cache_type
            if self.operator_step != 0
            else file_path.split(".")[-1]
        )

        if local_cache == "parquet":
            import pyarrow.parquet as pq

            count = pq.ParquetFile(file_path).metadata.num_rows
        elif local_cache in ["jsonl", "csv"]:
            count = 0
            with open(file_path, "rb") as handle:
                for _ in handle:
                    count += 1
            if local_cache == "csv":
                count -= 1  # drop the header row
        else:
            dataframe = self._load_local_file(file_path, local_cache)
            count = len(dataframe)

        self._cached_record_count = count
        return count

    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Any:
        if (
            getattr(self, "_current_streaming_chunk", None) is not None
        ):
            return self._convert_output(self._current_streaming_chunk, output_type)

        if self.cache_type not in ["jsonl", "csv"]:
            self.logger.warning(
                f"Current cache_type '{self.cache_type}' does not support optimized "
                "streaming. The storage will fall back to loading the full file "
                "into memory."
            )
        return super().read(output_type)

    def iter_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Yield the current step as a stream of frames, for the pipeline.

        With ``batch_size`` set, ``_load_local_file`` returns a chunk reader and
        each iteration yields one frame. With ``batch_size`` unset it returns a
        whole DataFrame, which is itself iterable over *column names* — so this
        branch yields column labels rather than frames. That is the reference's
        behavior and is reproduced rather than corrected; the batched pipelines
        only reach ``iter_chunks`` with a batch size set, so the quirk is
        unreachable from a generated pipeline.
        """
        file_path = self._get_cache_file_path(self.operator_step)
        local_cache = (
            self.cache_type
            if self.operator_step != 0
            else file_path.split(".")[-1]
        )
        reader = self._load_local_file(file_path, local_cache, self.batch_size)
        if hasattr(reader, "__iter__"):
            for chunk in reader:
                yield chunk
        else:
            yield reader
