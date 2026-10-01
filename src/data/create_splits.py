from pathlib import Path
from collections import defaultdict
import json
import random
import shutil


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


DATASET_ROOT = Path(
    "data/raw/garbage_classification/garbage_classification"
)

OUTPUT_ROOT = Path("data/splits")

DUPLICATE_REPORT = Path(
    "data/duplicate_report.json"
)


def collect_images():
    """Collect all image files."""

    return sorted(
        [
            path
            for path in DATASET_ROOT.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


def load_duplicate_groups():

    if not DUPLICATE_REPORT.exists():

        raise FileNotFoundError(
            "duplicate_report.json not found. "
            "Run find_duplicates.py first."
        )

    with DUPLICATE_REPORT.open(
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(file)

    groups = []

    for group in report["duplicate_groups"]:

        files = [
            Path(file)
            for file in group["files"]
        ]

        groups.append(files)

    return groups


def build_duplicate_mapping(
    duplicate_groups
):
    """
    Map every image to its duplicate group.
    Images not belonging to a duplicate group
    get their own individual group.
    """

    image_to_group = {}

    group_id = 0

    for group in duplicate_groups:

        for image in group:

            image_to_group[image] = group_id

        group_id += 1

    return image_to_group, group_id


def create_groups(
    image_files,
    image_to_group,
    next_group_id
):
    """
    Create leakage-safe groups.

    Every duplicate group stays together.
    Non-duplicate images become individual groups.
    """

    groups = defaultdict(list)

    current_group_id = next_group_id

    for image in image_files:

        if image in image_to_group:

            group_id = image_to_group[image]

        else:

            group_id = current_group_id

            current_group_id += 1

        groups[group_id].append(image)

    return list(groups.values())


def assign_groups(groups):

    random.seed(RANDOM_SEED)

    random.shuffle(groups)

    total_images = sum(
        len(group)
        for group in groups
    )

    target_train = total_images * TRAIN_RATIO
    target_val = total_images * VAL_RATIO

    train = []
    validation = []
    test = []

    train_count = 0
    val_count = 0

    for group in groups:

        group_size = len(group)

        if train_count + group_size <= target_train:

            train.extend(group)
            train_count += group_size

        elif val_count + group_size <= target_val:

            validation.extend(group)
            val_count += group_size

        else:

            test.extend(group)

    return train, validation, test


def copy_files(
    files,
    split_name
):

    split_root = (
        OUTPUT_ROOT / split_name
    )

    for image_path in files:

        class_name = image_path.parent.name

        destination_directory = (
            split_root / class_name
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            destination_directory
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


def count_classes(files):

    counts = defaultdict(int)

    for image in files:

        counts[image.parent.name] += 1

    return dict(
        sorted(counts.items())
    )


def main():

    print("=" * 70)
    print("LEAKAGE-SAFE DATASET SPLIT")
    print("=" * 70)

    image_files = collect_images()

    print(
        f"\nTotal images: "
        f"{len(image_files):,}"
    )

    duplicate_groups = (
        load_duplicate_groups()
    )

    print(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    image_to_group, next_group_id = (
        build_duplicate_mapping(
            duplicate_groups
        )
    )

    groups = create_groups(
        image_files,
        image_to_group,
        next_group_id
    )

    print(
        f"Total groups: "
        f"{len(groups):,}"
    )

    train, validation, test = (
        assign_groups(groups)
    )

    print("\nSplit sizes:")
    print("-" * 70)

    print(
        f"Train:       {len(train):,} "
        f"({len(train) / len(image_files) * 100:.2f}%)"
    )

    print(
        f"Validation:  {len(validation):,} "
        f"({len(validation) / len(image_files) * 100:.2f}%)"
    )

    print(
        f"Test:        {len(test):,} "
        f"({len(test) / len(image_files) * 100:.2f}%)"
    )

    print("\nCopying files...")

    copy_files(
        train,
        "train"
    )

    copy_files(
        validation,
        "validation"
    )

    copy_files(
        test,
        "test"
    )

    print("\nClass distribution:")
    print("-" * 70)

    print("\nTRAIN")

    for class_name, count in count_classes(train).items():

        print(
            f"{class_name:<20}{count:>6}"
        )

    print("\nVALIDATION")

    for class_name, count in count_classes(
        validation
    ).items():

        print(
            f"{class_name:<20}{count:>6}"
        )

    print("\nTEST")

    for class_name, count in count_classes(
        test
    ).items():

        print(
            f"{class_name:<20}{count:>6}"
        )

    print("\n" + "=" * 70)

    print(
        f"Dataset created at: "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()