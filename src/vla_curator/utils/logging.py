"""
Logging configuration.

Call ``setup_logging()`` once at the start of a script or pipeline run.
All modules use ``logging.getLogger(__name__)`` so the hierarchy is
automatically managed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    rich: bool = True,
) -> None:
    """
    Configure root logger with a console handler and optional file handler.

    Parameters
    ----------
    level : int or str
        Logging level (e.g. logging.INFO, "DEBUG").
    log_file : Path or None
        If provided, also write logs to this file.
    rich : bool
        Use rich.logging.RichHandler for prettier console output if available.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (idempotent)
    root.handlers.clear()

    # Console handler
    if rich:
        try:
            from rich.logging import RichHandler
            console_handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=False,
            )
        except ImportError:
            console_handler = _plain_console_handler()
    else:
        console_handler = _plain_console_handler()

    console_handler.setLevel(level)
    root.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _plain_console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — identical to logging.getLogger(name)."""
    return logging.getLogger(name)
