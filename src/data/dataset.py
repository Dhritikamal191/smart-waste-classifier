from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    MODEL_DIR,
)


def create_directories() -> None:
    """Create all required project directories."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODEL_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Project directories created successfully.")


if __name__ == "__main__":
    create_directories()