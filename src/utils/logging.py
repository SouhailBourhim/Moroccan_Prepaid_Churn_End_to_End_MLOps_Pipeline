from loguru import logger
import sys


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )
