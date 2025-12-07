"""Global logger configuration for the Silkroad project."""

import logging
import os
import sys

__all__ = ["logger", "setup_logger"]


def setup_logger(
    name: str = "silkroad",
    level: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically project name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    format_string = format_string or (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False

    return logger


# Create default logger instance for the project
logger = setup_logger()
