from pathlib import Path
from collections import defaultdict
import json


SPLITS = {
    "train": Path("data/splits/train"),
    "validation": Path("data/splits/validation"),
    "test": Path("data/splits/test"),
}

DUPLICATE_REPORT = Path(
    "data/duplicate_report.json"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


EXPECTED_CLASSES = {
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
}


def collect_files(directory):

    return {
        path.resolve()
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    }


def load_duplicate_groups():

    with DUPLICATE_REPORT.open(
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    return [
        {
            Path(file).resolve()
            for file in group["files"]
        }
        for group in report["duplicate_groups"]
    ]


def main():

    print("=" * 70)
    print("DATASET SPLIT VERIFICATION")
    print("=" * 70)

    split_files = {}

    # ---------------------------------------------------------
    # Check splits
    # ---------------------------------------------------------

    for split_name, split_path in SPLITS.items():

        if not split_path.exists():

            raise FileNotFoundError(
                f"Missing split: {split_path}"
            )

        files = collect_files(split_path)

        split_files[split_name] = files

        classes = {
            directory.name
            for directory in split_path.iterdir()
            if directory.is_dir()
        }

        missing_classes = (
            EXPECTED_CLASSES - classes
        )

        unexpected_classes = (
            classes - EXPECTED_CLASSES
        )

        print(f"\n{split_name.upper()}")

        print("-" * 70)

        print(
            f"Images: {len(files):,}"
        )

        print(
            f"Classes: {len(classes)}"
        )

        if missing_classes:

            print(
                f"Missing classes: "
                f"{sorted(missing_classes)}"
            )

        else:

            print(
                "All 12 classes present: PASS"
            )

        if unexpected_classes:

            print(
                f"Unexpected classes: "
                f"{sorted(unexpected_classes)}"
            )

    # ---------------------------------------------------------
    # Check overlap
    # ---------------------------------------------------------

    print("\n")
    print("SPLIT OVERLAP CHECK")
    print("-" * 70)

    train = split_files["train"]
    validation = split_files["validation"]
    test = split_files["test"]

    train_val = train & validation
    train_test = train & test
    val_test = validation & test

    print(
        f"Train ∩ Validation: "
        f"{len(train_val)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(val_test)}"
    )

    if (
        train_val
        or train_test
        or val_test
    ):

        print(
            "\nWARNING: File overlap detected."
        )

    else:

        print(
            "\nFile overlap check: PASS"
        )

    # ---------------------------------------------------------
    # Duplicate group leakage
    # ---------------------------------------------------------

    print("\n")
    print("DUPLICATE GROUP LEAKAGE CHECK")
    print("-" * 70)

    duplicate_groups = (
        load_duplicate_groups()
    )

    leakage_groups = []

    for index, group in enumerate(
        duplicate_groups,
        start=1,
    ):

        locations = set()

        for file in group:

            if file in train:
                locations.add("train")

            elif file in validation:
                locations.add("validation")

            elif file in test:
                locations.add("test")

        if len(locations) > 1:

            leakage_groups.append(
                {
                    "group": index,
                    "splits": sorted(locations),
                }
            )

    print(
        f"Duplicate groups checked: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Groups crossing splits: "
        f"{len(leakage_groups)}"
    )

    if leakage_groups:

        print("\nWARNING: Duplicate leakage detected.")

        for item in leakage_groups[:20]:

            print(
                f"Group {item['group']}: "
                f"{item['splits']}"
            )

    else:

        print(
            "Duplicate leakage check: PASS"
        )

    # ---------------------------------------------------------
    # Total
    # ---------------------------------------------------------

    total = sum(
        len(files)
        for files in split_files.values()
    )

    print("\n")
    print("TOTAL DATASET")
    print("-" * 70)

    print(
        f"Total split images: "
        f"{total:,}"
    )

    if total == 15_515:

        print(
            "Total image count check: PASS"
        )

    else:

        print(
            "WARNING: Total does not equal "
            "15,515."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()