from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/waste_classifier_finetuned.keras"
)

TEST_DIRECTORY = Path(
    "data/splits/test"
)

OUTPUT_DIRECTORY = Path(
    "models/evaluation/phase2"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# Load test dataset
# ============================================================

def load_test_dataset():

    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIRECTORY,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return dataset


# ============================================================
# Get predictions
# ============================================================

def get_predictions(model, dataset):

    y_true = []
    y_probability = []

    for images, labels in dataset:

        probabilities = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(
            labels.numpy()
        )

        y_probability.extend(
            probabilities
        )

    y_true = np.array(y_true)
    y_probability = np.array(y_probability)

    y_pred = np.argmax(
        y_probability,
        axis=1,
    )

    confidence = np.max(
        y_probability,
        axis=1,
    )

    return (
        y_true,
        y_pred,
        confidence,
    )


# ============================================================
# Analyze confusion pairs
# ============================================================

def analyze_confusion_pairs(
    y_true,
    y_pred,
    class_names,
):

    pairs = Counter()

    for true_label, predicted_label in zip(
        y_true,
        y_pred,
    ):

        if true_label != predicted_label:

            pair = (
                class_names[true_label],
                class_names[predicted_label],
            )

            pairs[pair] += 1

    return pairs


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 — ERROR ANALYSIS")
    print("=" * 70)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Phase 2 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = load_test_dataset()

    class_names = test_dataset.class_names

    print(
        f"Test images: "
        f"{len(test_dataset.file_paths)}"
    )

    print(
        f"Classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    (
        y_true,
        y_pred,
        confidence,
    ) = get_predictions(
        model,
        test_dataset,
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    incorrect = (
        y_true != y_pred
    )

    error_count = np.sum(
        incorrect
    )

    accuracy = (
        1
        - error_count / len(y_true)
    )

    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nTotal test images: {len(y_true):,}"
    )

    print(
        f"Correct predictions: "
        f"{len(y_true) - error_count:,}"
    )

    print(
        f"Incorrect predictions: "
        f"{error_count:,}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Error rate: "
        f"{(1 - accuracy) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion pairs
    # --------------------------------------------------------

    pairs = analyze_confusion_pairs(
        y_true,
        y_pred,
        class_names,
    )

    print("\n")
    print("=" * 70)
    print("TOP CONFUSION PAIRS")
    print("=" * 70)

    print(
        "\nTrue Class → Predicted Class"
    )

    print("-" * 70)

    for (
        (true_class, predicted_class),
        count,
    ) in pairs.most_common():

        print(
            f"{true_class:20s} → "
            f"{predicted_class:20s} "
            f"{count:4d}"
        )

    # --------------------------------------------------------
    # Per-class errors
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ERRORS BY TRUE CLASS")
    print("=" * 70)

    print("-" * 70)

    for class_index, class_name in enumerate(
        class_names
    ):

        class_mask = (
            y_true == class_index
        )

        class_errors = np.sum(
            y_pred[class_mask]
            != class_index
        )

        class_total = np.sum(
            class_mask
        )

        class_error_rate = (
            class_errors / class_total
            if class_total > 0
            else 0
        )

        print(
            f"{class_name:20s} "
            f"errors={class_errors:4d} "
            f"/ {class_total:4d} "
            f"rate={class_error_rate * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Most confident mistakes
    # --------------------------------------------------------

    wrong_indices = np.where(
        incorrect
    )[0]

    print("\n")
    print("=" * 70)
    print("MOST CONFIDENT MISTAKES")
    print("=" * 70)

    if len(wrong_indices) > 0:

        sorted_indices = wrong_indices[
            np.argsort(
                confidence[wrong_indices]
            )[::-1]
        ]

        print(
            "\nTop 20:"
        )

        print("-" * 70)

        for index in sorted_indices[:20]:

            true_class = class_names[
                y_true[index]
            ]

            predicted_class = class_names[
                y_pred[index]
            ]

            print(
                f"\nTrue:       {true_class}"
            )

            print(
                f"Predicted:  {predicted_class}"
            )

            print(
                f"Confidence: "
                f"{confidence[index]:.4f}"
            )

            print(
                f"File:       "
                f"{test_dataset.file_paths[index]}"
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIRECTORY
        / "error_analysis.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHASE 3 — ERROR ANALYSIS\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Total test images: {len(y_true)}\n"
        )

        file.write(
            f"Correct predictions: "
            f"{len(y_true) - error_count}\n"
        )

        file.write(
            f"Incorrect predictions: "
            f"{error_count}\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy * 100:.4f}%\n"
        )

        file.write(
            f"Error rate: "
            f"{(1 - accuracy) * 100:.4f}%\n\n"
        )

        file.write(
            "CONFUSION PAIRS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for (
            (true_class, predicted_class),
            count,
        ) in pairs.most_common():

            file.write(
                f"{true_class} -> "
                f"{predicted_class}: "
                f"{count}\n"
            )

        file.write(
            "\nERRORS BY TRUE CLASS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for class_index, class_name in enumerate(
            class_names
        ):

            class_mask = (
                y_true == class_index
            )

            class_errors = np.sum(
                y_pred[class_mask]
                != class_index
            )

            class_total = np.sum(
                class_mask
            )

            file.write(
                f"{class_name}: "
                f"{class_errors}/{class_total}\n"
            )

    print("\n")
    print("=" * 70)
    print("PHASE 3 ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nReport saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()