from pathlib import Path

import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("data/splits")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Dataset creation
# ============================================================

def load_dataset(
    directory,
    shuffle,
):
    """
    Load an image dataset from a directory structure.

    Expected structure:

        directory/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                image3.jpg
                ...

    Returns:
        tf.data.Dataset
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation",
)


# ============================================================
# Prepare datasets
# ============================================================

def prepare_train_dataset(dataset):

    dataset = dataset.map(
        lambda images, labels: (
            data_augmentation(images, training=True),
            labels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


def prepare_eval_dataset(dataset):

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


# ============================================================
# Main pipeline
# ============================================================

def create_datasets():

    train_directory = (
        DATA_ROOT / "train"
    )

    validation_directory = (
        DATA_ROOT / "validation"
    )

    test_directory = (
        DATA_ROOT / "test"
    )

    train_dataset = load_dataset(
        train_directory,
        shuffle=True,
    )

    validation_dataset = load_dataset(
        validation_directory,
        shuffle=False,
    )

    test_dataset = load_dataset(
        test_directory,
        shuffle=False,
    )

    class_names = (
        train_dataset.class_names
    )

    train_dataset = prepare_train_dataset(
        train_dataset
    )

    validation_dataset = prepare_eval_dataset(
        validation_dataset
    )

    test_dataset = prepare_eval_dataset(
        test_dataset
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = create_datasets()

    print("=" * 70)
    print("TENSORFLOW DATASET PIPELINE")
    print("=" * 70)

    print(
        "\nClasses:"
    )

    for index, name in enumerate(
        class_names
    ):

        print(
            f"{index}: {name}"
        )

    print("\nChecking a training batch...")

    for images, labels in train_dataset.take(1):

        print(
            f"Image batch shape: "
            f"{images.shape}"
        )

        print(
            f"Label batch shape: "
            f"{labels.shape}"
        )

        print(
            f"Image dtype: "
            f"{images.dtype}"
        )

        print(
            f"Image minimum: "
            f"{tf.reduce_min(images).numpy():.2f}"
        )

        print(
            f"Image maximum: "
            f"{tf.reduce_max(images).numpy():.2f}"
        )

    print("\nChecking validation batch...")

    for images, labels in validation_dataset.take(1):

        print(
            f"Validation image shape: "
            f"{images.shape}"
        )

        print(
            f"Validation label shape: "
            f"{labels.shape}"
        )

    print("\nChecking test batch...")

    for images, labels in test_dataset.take(1):

        print(
            f"Test image shape: "
            f"{images.shape}"
        )

        print(
            f"Test label shape: "
            f"{labels.shape}"
        )

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)