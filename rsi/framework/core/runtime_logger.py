"""Logger for the dataflow-compatible runtime.

The reference runtime (``open-dataflow`` 1.0.10) logs through a logger that
carries a custom ``SUCCESS`` level between ``INFO`` and ``WARNING``, and calls
``logger.success(...)`` when a storage step is written. Reproducing that level
keeps generated operators and pipelines source-compatible: code written against
the reference may call ``self.logger.success(...)`` from an operator body.

Log text and formatting are presentation, not execution semantics; the parity
suite compares behavior, not log output.
"""

from __future__ import annotations

import logging
import sys

SUCCESS_LEVEL_NUM = 25
LOGGER_NAME = "rsi.framework.dataflow"

_CONFIGURED = False


def _install_success_level() -> None:
    if logging.getLevelName(SUCCESS_LEVEL_NUM) != "SUCCESS":
        logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
    if not hasattr(logging.Logger, "success"):
        def success(self: logging.Logger, message: str, *args, **kwargs) -> None:
            if self.isEnabledFor(SUCCESS_LEVEL_NUM):
                self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

        logging.Logger.success = success  # type: ignore[attr-defined]


def get_logger() -> logging.Logger:
    """Return the shared runtime logger, configured once per interpreter."""
    global _CONFIGURED
    _install_success_level()
    logger = logging.getLogger(LOGGER_NAME)
    if not _CONFIGURED:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | DataFlow | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        _CONFIGURED = True
    return logger
