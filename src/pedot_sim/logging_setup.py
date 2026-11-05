"""
Logging configuration with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "pedot_sim",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_file : Path, optional
        Path to log file. If None, no file handler is added.
    console : bool
        Whether to add a console handler.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
