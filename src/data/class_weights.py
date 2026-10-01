from pathlib import Path
from collections import Counter
import json


TRAIN_DIRECTORY = Path(
    "data/splits/train"
)

OUTPUT_FILE = Path(
    "data/class_weights.json"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_class_counts():

    counts = Counter()

    for class_directory in TRAIN_DIRECTORY.iterdir():

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


def calculate_weights(counts):

    total = sum(counts.values())
    number_of_classes = len(counts)

    weights = {}

    for class_name, count in counts.items():

        weights[class_name] = (
            total
            / (
                number_of_classes
                * count
            )
        )

    return weights


def main():

    print("=" * 70)
    print("TRAINING CLASS WEIGHTS")
    print("=" * 70)

    counts = get_class_counts()

    weights = calculate_weights(
        counts
    )

    print(
        f"\nTraining images: "
        f"{sum(counts.values()):,}"
    )

    print(
        f"Classes: "
        f"{len(counts)}"
    )

    print("\nClass weights:")
    print("-" * 70)

    for class_name in sorted(weights):

        print(
            f"{class_name:<20}"
            f"{counts[class_name]:>6} images   "
            f"weight={weights[class_name]:.4f}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keras expects integer class indices.
    # The mapping follows TensorFlow's alphabetical
    # class ordering.

    class_names = sorted(counts)

    class_weights = {
        str(index): weights[class_name]
        for index, class_name
        in enumerate(class_names)
    }

    output = {
        "class_names": class_names,
        "class_counts": dict(
            sorted(counts.items())
        ),
        "class_weights": class_weights,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()