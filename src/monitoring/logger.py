import logging
import sys


def configure_logging() -> logging.Logger:

    logger = logging.getLogger(
        "smart_waste_classifier"
    )

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(
        sys.stdout
    )

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False

    return logger


logger = configure_logging()