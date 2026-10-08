"""Bind one explicit structured input file and build a best-effort prompt preview."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from rsi.framework.evolution.utils.logging import get_logger

logger = get_logger("core.input_corpus")

# 入口发现可识别的结构化格式；生成算子须按实际格式解析。
SUPPORTED_ENTRY_SUFFIXES = (".jsonl", ".json", ".csv", ".parquet")

PREVIEW_ROWS = 3            # 预览行数：够看清字段与风格，又不至于挤占 prompt
PREVIEW_VALUE_MAXLEN = 200  # 单字段截断长度：长文档字段（如整篇正文）不该淹没其它字段
ROW_COUNT_CAP = 200_000     # 行数统计上限：超大语料只报「≥上限」，避免启动时全量扫盘
JSON_PARSE_MAX_BYTES = 32 * 1024 * 1024   # 整体式 .json 超过此大小不解析，只报文件信息


@dataclass(frozen=True)
class CorpusFile:
    """输入文件及其尽力而为的预览。"""

    path: Path
    rel: str                            # 相对输入根目录的路径，用于展示
    size_bytes: int
    rows: int | None = None             # None = 未知（不可解析格式或解析失败）
    rows_capped: bool = False           # True 表示行数统计触顶，实际 ≥ rows
    fields: tuple[str, ...] = ()
    preview_lines: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [self.rel, _human_size(self.size_bytes)]
        if self.rows is not None:
            parts.append(f"{'≥' if self.rows_capped else ''}{self.rows} 行")
        if self.fields:
            parts.append("字段: " + ", ".join(self.fields))
        return "  ".join(parts)


@dataclass(frozen=True)
class InputCorpus:
    """一次运行的固定输入契约：唯一入口文件及其 prompt 预览。"""

    entry_path: str
    root: Path | None = None
    files: tuple[CorpusFile, ...] = ()
    entry_file: CorpusFile | None = field(default=None)

    @property
    def grounded(self) -> bool:
        """是否已绑定真实语料。轻量入口要求此值始终为真。"""
        return self.entry_file is not None

    def sample_text(self) -> str | None:
        """返回入口文件的前几行。"""
        if self.entry_file is not None and self.entry_file.preview_lines:
            return "\n".join(self.entry_file.preview_lines)
        return None

    def prompt_block(self) -> str:
        """渲染注入 PipelineAgent prompt 的输入语料段。"""
        if not self.grounded:
            return ""
        lines = [f"\n【输入语料】固定 Entry file：{self.entry_path}"]
        if self.entry_file is not None:
            lines.append(f"  - {self.entry_file.describe()}")
        preview = self.entry_file.preview_lines if self.entry_file else ()
        if preview:
            lines.append(f"  Entry file 前 {len(preview)} 行（长字段已截断，仅供了解字段与风格）：")
            lines.extend(f"    {line}" for line in preview)
        return "\n".join(lines) + "\n"


def discover(input_path: str | Path | None) -> InputCorpus:
    """Bind the required structured entry file without directory discovery."""
    if not input_path:
        raise ValueError("project.input_path 不能为空：review-only loop 需要固定 raw data")
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"raw data entry file 不存在或不是文件：{path}")
    if path.suffix.lower() not in SUPPORTED_ENTRY_SUFFIXES:
        raise ValueError(
            f"project.input_path 必须指向结构化文件：{', '.join(SUPPORTED_ENTRY_SUFFIXES)}；"
            f"实际为 {path.suffix or '<无扩展名>'}"
        )

    entry = _profile_file(path, path.name, path.stat().st_size)
    logger.info("固定 raw corpus：Entry file=%s（%s）。", path, entry.describe())
    return InputCorpus(
        entry_path=str(path), root=path.parent, files=(entry,), entry_file=entry
    )


# ---- 各格式的尽力而为预览 --------------------------------------------------
def _profile_file(path: Path, rel: str, size: int) -> CorpusFile:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return _profile_jsonl(path, rel, size)
        if suffix == ".json":
            return _profile_json(path, rel, size)
        if suffix == ".csv":
            return _profile_csv(path, rel, size)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("预览输入文件 %s 失败（%s），仅按文件名列出。", rel, exc)
    return CorpusFile(path=path, rel=rel, size_bytes=size)


def _profile_jsonl(path: Path, rel: str, size: int) -> CorpusFile:
    rows = 0
    capped = False
    sample: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            if len(sample) < PREVIEW_ROWS:
                obj = _loads(line)
                if obj is not None:
                    sample.append(obj)
            if rows >= ROW_COUNT_CAP:
                capped = True
                break
    return _build(path, rel, size, rows, capped, sample)


def _profile_json(path: Path, rel: str, size: int) -> CorpusFile:
    if size > JSON_PARSE_MAX_BYTES:
        logger.info("输入文件 %s 过大（%s），跳过预览解析。", rel, _human_size(size))
        return CorpusFile(path=path, rel=rel, size_bytes=size)
    obj = _loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, list):
        sample = [r for r in obj[:PREVIEW_ROWS] if isinstance(r, dict)]
        return _build(path, rel, size, len(obj), False, sample)
    if isinstance(obj, dict):
        return _build(path, rel, size, 1, False, [obj])
    return CorpusFile(path=path, rel=rel, size_bytes=size)


def _profile_csv(path: Path, rel: str, size: int) -> CorpusFile:
    rows = 0
    capped = False
    sample: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            if len(sample) < PREVIEW_ROWS:
                sample.append({k: v for k, v in row.items() if k is not None})
            if rows >= ROW_COUNT_CAP:
                capped = True
                break
    return _build(path, rel, size, rows, capped, sample)


def _build(
    path: Path, rel: str, size: int, rows: int, capped: bool, sample: list[dict]
) -> CorpusFile:
    fields: list[str] = []
    for row in sample:                      # 并集而非首行：稀疏字段也该被 Agent 看见
        for k in row:
            if k not in fields:
                fields.append(str(k))
    preview = tuple(
        json.dumps({str(k): _truncate_value(v) for k, v in row.items()}, ensure_ascii=False)
        for row in sample
    )
    return CorpusFile(
        path=path, rel=rel, size_bytes=size, rows=rows, rows_capped=capped,
        fields=tuple(fields), preview_lines=preview,
    )


def _loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _truncate_value(value):
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        return [_truncate_value(v) for v in value[:3]]
    if isinstance(value, dict):
        return {str(k): _truncate_value(v) for k, v in list(value.items())[:5]}
    return value


def _truncate(text: str) -> str:
    if len(text) <= PREVIEW_VALUE_MAXLEN:
        return text
    return text[:PREVIEW_VALUE_MAXLEN] + f"…(共 {len(text)} 字符)"


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"
