import logging
from typing import Literal

# Define allowed log level names
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Map string levels to logging constants
_LEVEL_MAP: dict[str, int] = {
    "NOTSET": logging.NOTSET,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def create_logger(name: str, level: LogLevel = "INFO") -> logging.Logger:
    """
    Create a logger with the specified name and level.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(_LEVEL_MAP[level])
    logger.propagate = False

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger