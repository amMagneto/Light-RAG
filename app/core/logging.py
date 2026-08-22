import logging
import sys

_CONFIGURED = False

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFMT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
