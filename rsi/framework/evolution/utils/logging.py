"""统一的结构化日志：带节点/阶段上下文前缀，输出到控制台与文件。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """配置根日志器。重复调用是幂等的（只配置一次）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("rsi.framework.evolution")
    root.setLevel(level.upper())
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """返回 rsi.framework.evolution.<name> 命名空间下的日志器。"""
    return logging.getLogger(f"rsi.framework.evolution.{name}")
