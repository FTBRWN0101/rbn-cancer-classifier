import logging
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger.

    Parameters
    name : str: Logger name.
    level : int: Logging level.
    Returns
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
