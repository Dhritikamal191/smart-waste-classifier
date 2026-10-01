from pathlib import Path
from collections import defaultdict
import json
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


HASH_SIZE = 16


def collect_images(dataset_root: Path):
    """Collect all supported image files."""

    return sorted(
        [
            path
            for path in dataset_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )


def calculate_average_hash(image_path: Path):
    """
    Calculate a perceptual average hash using
    Pillow only.

    This implementation intentionally avoids NumPy
    and ImageHash because some native extensions are
    being blocked by the Windows application-control
    policy.
    """

    try:
        with Image.open(image_path) as image:

            image = image.convert("L")

            image = ImageOps.fit(
                image,
                (HASH_SIZE, HASH_SIZE),
            )

            pixels = list(image.getdata())

            average = sum(pixels) / len(pixels)

            bits = [
                "1" if pixel >= average else "0"
                for pixel in pixels
            ]

            return "".join(bits)

    except Exception as error:

        print(
            f"Could not hash {image_path}: {error}"
        )

        return None


def find_duplicates(image_files):
    """Find images with identical perceptual hashes."""

    hash_to_files = defaultdict(list)

    failed = []

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        image_hash = calculate_average_hash(
            image_path
        )

        if image_hash is not None:

            hash_to_files[image_hash].append(
                image_path
            )

        else:

            failed.append(image_path)

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index:,}/{len(image_files):,} images..."
            )

    duplicate_groups = {
        image_hash: files
        for image_hash, files in hash_to_files.items()
        if len(files) > 1
    }

    report = {
    "total_images": len(image_files),
    "failed_hashes": len(failed),
    "duplicate_groups": [
        {
            "hash": image_hash,
            "files": [
                str(file)
                for file in files
            ],
        }
        for image_hash, files
        in duplicate_groups.items()
        ],
    }

    report_path = Path(
    "data/duplicate_report.json"
    )

    report_path.parent.mkdir(
    parents=True,
    exist_ok=True
    )

    with report_path.open(
    "w",
    encoding="utf-8"
) as file:

        json.dump(
        report,
        file,
        indent=2
        )

    print(
    f"\nDuplicate report saved to: "
    f"{report_path}"
)
    
    return duplicate_groups, failed


def main():

    dataset_root = Path(
        "data/raw/garbage_classification/"
        "garbage_classification"
    )

    image_files = collect_images(
        dataset_root
    )

    print(
        f"Found {len(image_files):,} images."
    )

    duplicate_groups, failed = find_duplicates(
        image_files
    )

    print("\n")
    print("=" * 70)
    print("DUPLICATE IMAGE ANALYSIS")
    print("=" * 70)

    print(
        f"\nImages processed successfully: "
        f"{len(image_files) - len(failed):,}"
    )

    print(
        f"Images that failed hashing: "
        f"{len(failed):,}"
    )

    print(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    duplicate_image_count = sum(
        len(files)
        for files in duplicate_groups.values()
    )

    print(
        f"Images involved in duplicate groups: "
        f"{duplicate_image_count}"
    )

    if failed:

        print("\nWARNING:")
        print(
            "Some images could not be hashed."
        )

        for file in failed[:10]:
            print(f"  {file}")

    if duplicate_groups:

        print("\nDuplicate groups:")
        print("-" * 70)

        for index, (
            image_hash,
            files,
        ) in enumerate(
            duplicate_groups.items(),
            start=1,
        ):

            print(
                f"\nGroup {index} "
                f"(hash={image_hash})"
            )

            for file in files:
                print(f"  {file}")

            if index >= 30:

                print(
                    "\nOnly the first 30 groups "
                    "are displayed."
                )

                break

    else:

        print(
            "\nNo duplicate groups found."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()