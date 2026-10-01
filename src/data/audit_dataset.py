from pathlib import Path
from collections import Counter

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images(dataset_root: Path):
    """Collect all supported image files."""

    image_files = []

    for path in dataset_root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            image_files.append(path)

    return sorted(image_files)


def check_images(image_files):
    """Check images for corruption and collect dimensions."""

    corrupted = []
    dimensions = Counter()

    for index, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as image:

                # Verify image integrity.
                image.verify()

            # Re-open after verify because verify() invalidates
            # the image object.
            with Image.open(image_path) as image:

                dimensions[image.size] += 1

        except Exception as error:

            corrupted.append(
                {
                    "path": str(image_path),
                    "error": str(error),
                }
            )

        if index % 1000 == 0:
            print(
                f"Checked {index:,}/{len(image_files):,} images..."
            )

    return corrupted, dimensions


def class_distribution(image_files):
    """Calculate image count per class."""

    counts = Counter()

    for image_path in image_files:

        class_name = image_path.parent.name

        counts[class_name] += 1

    return counts


def print_report(
    image_files,
    corrupted,
    dimensions,
    class_counts,
):
    """Print the dataset audit report."""

    print("\n")
    print("=" * 70)
    print("DATASET QUALITY AUDIT")
    print("=" * 70)

    print(f"\nTotal images: {len(image_files):,}")

    print("\nCorrupted images:")
    print("-" * 70)

    if corrupted:
        print(f"Found: {len(corrupted)}")

        for item in corrupted[:20]:
            print(item["path"])
            print(f"Reason: {item['error']}")

    else:
        print("None found.")

    print("\nImage dimensions:")
    print("-" * 70)

    for dimension, count in dimensions.most_common(15):
        print(
            f"{str(dimension):<15}"
            f"{count:>7} images"
        )

    print("\nClass distribution:")
    print("-" * 70)

    for class_name, count in class_counts.most_common():
        percentage = (
            count / len(image_files)
        ) * 100

        print(
            f"{class_name:<20}"
            f"{count:>7} "
            f"({percentage:>6.2f}%)"
        )

    print("\n" + "=" * 70)


def main():

    dataset_root = Path(
        "data/raw/garbage_classification/garbage_classification"
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    print(f"Dataset: {dataset_root}")

    image_files = collect_images(dataset_root)

    print(
        f"Found {len(image_files):,} image files."
    )

    corrupted, dimensions = check_images(
        image_files
    )

    class_counts = class_distribution(
        image_files
    )

    print_report(
        image_files,
        corrupted,
        dimensions,
        class_counts,
    )


if __name__ == "__main__":
    main()