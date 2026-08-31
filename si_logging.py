#!/usr/bin/env python3
"""Structured, industrial logging for sportintel_core."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from si_config import LOG_JSON, LOG_LEVEL


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("step", "fixture", "team", "duration_s", "n_ok", "n_fail"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "sportintel") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_step(logger: logging.Logger, step: str, msg: str, **extra: Any) -> None:
    extra_fields = {"step": step, **extra}
    logger.info(msg, extra=extra_fields)


def log_exception(
    logger: logging.Logger,
    msg: str,
    exc: Optional[BaseException] = None,
    *,
    critical: bool = False,
) -> None:
    """
    Always log the exception. Use critical=True for failures that should
    surface loudly in CI without necessarily aborting the whole process.
    """
    level = logging.CRITICAL if critical else logging.ERROR
    logger.log(level, msg, exc_info=exc or True)
