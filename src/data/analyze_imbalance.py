from pathlib import Path
from collections import Counter

import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_class_counts(dataset_root: Path):

    counts = Counter()

    for class_directory in dataset_root.iterdir():

        if not class_directory.is_dir():
            continue

        count = sum(
            1
            for file in class_directory.rglob("*")
            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        counts[class_directory.name] = count

    return counts


def calculate_class_weights(counts):

    total = sum(counts.values())
    number_of_classes = len(counts)

    weights = {}

    for class_name, count in counts.items():

        weights[class_name] = (
            total
            / (number_of_classes * count)
        )

    return weights


def main():

    dataset_root = Path(
        "data/raw/garbage_classification/garbage_classification"
    )

    counts = get_class_counts(
        dataset_root
    )

    weights = calculate_class_weights(
        counts
    )

    print("=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal images: "
        f"{sum(counts.values()):,}"
    )

    print(
        f"Number of classes: "
        f"{len(counts)}"
    )

    print("\nClass statistics:")
    print("-" * 70)

    minimum = min(counts.values())
    maximum = max(counts.values())

    print(
        f"Minimum class size: {minimum:,}"
    )

    print(
        f"Maximum class size: {maximum:,}"
    )

    print(
        f"Imbalance ratio: "
        f"{maximum / minimum:.2f}x"
    )

    print("\nCalculated class weights:")
    print("-" * 70)

    for class_name in sorted(weights):

        print(
            f"{class_name:<20}"
            f"{counts[class_name]:>6} images  "
            f"weight={weights[class_name]:.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()