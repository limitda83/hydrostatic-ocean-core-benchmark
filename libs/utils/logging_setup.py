#########################################################################
#  Module: logging_setup                                                #
#  Description: Configure the stdlib logging for console and per-run    #
#               log files. print() is never used for operational output.#
#  Pipeline: entry point -> logging_setup -> all modules                #
#########################################################################

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Install console (and optionally file) handlers on the root logger."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(file_handler)

    return root
