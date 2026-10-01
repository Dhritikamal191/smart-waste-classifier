from pathlib import Path
from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_dataset_root(base_dir: Path) -> Path:
    """
    Find the directory containing the class folders.
    """

    candidates = []

    for path in base_dir.rglob("*"):
        if not path.is_dir():
            continue

        subdirectories = [
            item for item in path.iterdir()
            if item.is_dir()
        ]

        if len(subdirectories) >= 2:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset directory inside {base_dir}"
        )

    # Prefer the directory closest to the extracted dataset root.
    return sorted(
        candidates,
        key=lambda p: len(p.parts)
    )[0]


def inspect_dataset(dataset_root: Path) -> None:
    """
    Inspect class names and image counts.
    """

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    class_directories = sorted(
        [
            directory
            for directory in dataset_root.iterdir()
            if directory.is_dir()
        ]
    )

    if not class_directories:
        raise ValueError(
            f"No class directories found in {dataset_root}"
        )

    total_images = 0

    print(f"\nDataset root: {dataset_root}")
    print(f"Number of classes: {len(class_directories)}\n")

    print("Class distribution:")
    print("-" * 60)

    for class_directory in class_directories:

        image_files = [
            file
            for file in class_directory.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = len(image_files)

        total_images += count

        print(
            f"{class_directory.name:<20} "
            f"{count:>6} images"
        )

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_images:>6} images")


def main() -> None:

    base_dir = Path("data/raw")

    dataset_root = find_dataset_root(base_dir)

    inspect_dataset(dataset_root)


if __name__ == "__main__":
    main()