"""Logging setup shared by the CLI and the server."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE = "%H:%M:%S"


class _ColourFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        colour = self.COLOURS.get(record.levelno)
        return f"{colour}{text}{self.RESET}" if colour else text


def setup_logging(level: str = "INFO", colour: bool | None = None) -> None:
    """Configure root logging once, with colour when writing to a terminal."""
    if colour is None:
        colour = sys.stderr.isatty()
    handler = logging.StreamHandler(sys.stderr)
    formatter = (_ColourFormatter if colour else logging.Formatter)(_FORMAT, _DATE)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # OpenCV and urllib3 are chatty at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
