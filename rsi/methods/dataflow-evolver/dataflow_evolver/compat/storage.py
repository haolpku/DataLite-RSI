"""Process-local compatibility fixes for open-dataflow storage semantics."""
from __future__ import annotations

from typing import Any

from dataflow.utils.storage import FileStorage


_INSTALL_MARKER = "_dfe_null_normalization_installed"
_ORIGINAL_CONVERT_ATTR = "_dfe_original_convert_output"


def dataframe_records_with_python_nulls(dataframe: Any) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to records without leaking pandas missing values.

    ``FileStorage`` reads JSONL through ``pandas.read_json``. JSON ``null`` values
    therefore become ``NaN``/``NaT``/``pd.NA`` before ``to_dict`` is called. Those
    values are truthy or otherwise unsafe in ordinary operator fallback expressions.
    JSON has only one missing-value representation, so restore every top-level
    pandas missing cell to Python ``None`` at the ``read("dict")`` boundary.
    """

    normalized = dataframe.astype(object).where(dataframe.notna(), None)
    return normalized.to_dict(orient="records")


def install_file_storage_null_normalization() -> bool:
    """Patch ``FileStorage.read('dict')`` semantics for this pipeline process.

    The installed open-dataflow package remains untouched. The patch is idempotent
    and deliberately leaves DataFrame reads unchanged, where pandas missing-value
    semantics are expected.
    """

    storage_cls = FileStorage
    if getattr(storage_cls, _INSTALL_MARKER, False):
        return False

    original_convert = storage_cls._convert_output
    setattr(storage_cls, _ORIGINAL_CONVERT_ATTR, original_convert)

    def convert_output(self, dataframe: Any, output_type: str) -> Any:
        if output_type == "dict":
            return dataframe_records_with_python_nulls(dataframe)
        return original_convert(self, dataframe, output_type)

    storage_cls._convert_output = convert_output
    setattr(storage_cls, _INSTALL_MARKER, True)
    return True
