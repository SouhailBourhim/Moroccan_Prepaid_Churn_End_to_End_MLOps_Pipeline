from __future__ import annotations

import os
import sys

from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    if os.environ.get("LOG_FORMAT") == "json":
        # Structured JSON output for production log aggregation (Datadog, CloudWatch, etc.)
        logger.add(sys.stderr, serialize=True, level=level)
    else:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            level=level,
            colorize=True,
        )
    logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )
