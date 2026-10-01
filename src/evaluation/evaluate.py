from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

def get_predictions(
    model,
    dataset,
):

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

    y_true = np.array(
        y_true
    )

    y_probability = np.array(
        y_probability
    )

    y_pred = np.argmax(
        y_probability,
        axis=1,
    )

    return (
        y_true,
        y_pred,
        y_probability,
    )


# ============================================================
# Calculate confusion matrix
# ============================================================

def calculate_confusion_matrix(
    y_true,
    y_pred,
    num_classes,
):

    matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        y_true,
        y_pred,
    ):

        matrix[
            true_label,
            predicted_label,
        ] += 1

    return matrix


# ============================================================
# Calculate classification metrics
# ============================================================

def calculate_classification_metrics(
    matrix,
):

    metrics = []

    for class_id in range(
        matrix.shape[0]
    ):

        true_positive = matrix[
            class_id,
            class_id,
        ]

        false_positive = (
            matrix[:, class_id].sum()
            - true_positive
        )

        false_negative = (
            matrix[class_id, :].sum()
            - true_positive
        )

        support = matrix[
            class_id,
            :
        ].sum()

        # Precision
        precision_denominator = (
            true_positive
            + false_positive
        )

        if precision_denominator > 0:

            precision = (
                true_positive
                / precision_denominator
            )

        else:

            precision = 0.0

        # Recall
        recall_denominator = (
            true_positive
            + false_negative
        )

        if recall_denominator > 0:

            recall = (
                true_positive
                / recall_denominator
            )

        else:

            recall = 0.0

        # F1 score
        if (
            precision + recall
            > 0
        ):

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        metrics.append(
            {
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(
                    f1
                ),
                "support": int(
                    support
                ),
            }
        )

    return metrics


# ============================================================
# Print classification report
# ============================================================

def print_classification_report(
    class_names,
    class_metrics,
):

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        f"{'Class':<18}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
        f"{'Support':>12}"
    )

    print("-" * 70)

    for class_name, metrics in zip(
        class_names,
        class_metrics,
    ):

        print(
            f"{class_name:<18}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['support']:>12}"
        )

    # --------------------------------------------------------
    # Macro averages
    # --------------------------------------------------------

    macro_precision = np.mean(
        [
            metric["precision"]
            for metric in class_metrics
        ]
    )

    macro_recall = np.mean(
        [
            metric["recall"]
            for metric in class_metrics
        ]
    )

    macro_f1 = np.mean(
        [
            metric["f1"]
            for metric in class_metrics
        ]
    )

    # --------------------------------------------------------
    # Weighted averages
    # --------------------------------------------------------

    total_support = sum(
        metric["support"]
        for metric in class_metrics
    )

    if total_support > 0:

        weighted_precision = (
            sum(
                metric["precision"]
                * metric["support"]
                for metric in class_metrics
            )
            / total_support
        )

        weighted_recall = (
            sum(
                metric["recall"]
                * metric["support"]
                for metric in class_metrics
            )
            / total_support
        )

        weighted_f1 = (
            sum(
                metric["f1"]
                * metric["support"]
                for metric in class_metrics
            )
            / total_support
        )

    else:

        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    # --------------------------------------------------------
    # Print averages
    # --------------------------------------------------------

    print("-" * 70)

    print(
        f"{'Macro avg':<18}"
        f"{macro_precision:>12.4f}"
        f"{macro_recall:>12.4f}"
        f"{macro_f1:>12.4f}"
        f"{total_support:>12}"
    )

    print(
        f"{'Weighted avg':<18}"
        f"{weighted_precision:>12.4f}"
        f"{weighted_recall:>12.4f}"
        f"{weighted_f1:>12.4f}"
        f"{total_support:>12}"
    )

    return {
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }


# ============================================================
# Save confusion matrix
# ============================================================

def save_confusion_matrix(
    matrix,
    class_names,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_size = (
        max(
            10,
            len(class_names) * 0.8,
        ),
        max(
            8,
            len(class_names) * 0.7,
        ),
    )

    plt.figure(
        figsize=figure_size
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "Waste Classifier — Phase 2 Test Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(class_names)
    )

    plt.xticks(
        tick_positions,
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_positions,
        class_names,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    # --------------------------------------------------------
    # Write values inside cells
    # --------------------------------------------------------

    threshold = (
        matrix.max() / 2
        if matrix.max() > 0
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column,
            ]

            plt.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================
# Save evaluation summary
# ============================================================

def save_evaluation_summary(
    test_loss,
    test_accuracy,
    classification_metrics,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WASTE CLASSIFIER — PHASE 2 EVALUATION\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Test Loss: "
            f"{test_loss:.6f}\n"
        )

        file.write(
            f"Test Accuracy: "
            f"{test_accuracy:.6f}\n"
        )

        file.write(
            f"Test Accuracy (%): "
            f"{test_accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "Macro Precision: "
            f"{classification_metrics['macro_precision']:.6f}\n"
        )

        file.write(
            "Macro Recall: "
            f"{classification_metrics['macro_recall']:.6f}\n"
        )

        file.write(
            "Macro F1: "
            f"{classification_metrics['macro_f1']:.6f}\n\n"
        )

        file.write(
            "Weighted Precision: "
            f"{classification_metrics['weighted_precision']:.6f}\n"
        )

        file.write(
            "Weighted Recall: "
            f"{classification_metrics['weighted_recall']:.6f}\n"
        )

        file.write(
            "Weighted F1: "
            f"{classification_metrics['weighted_f1']:.6f}\n"
        )

    return summary_path


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 2 TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading Phase 2 model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print(
        "\nLoading untouched test dataset..."
    )

    test_dataset = load_test_dataset()

    class_names = (
        test_dataset.class_names
    )

    print(
        f"Test images: "
        f"{len(test_dataset.file_paths)}"
    )

    print(
        f"Classes: "
        f"{len(class_names)}"
    )

    print(
        f"Class names: "
        f"{class_names}"
    )

    # --------------------------------------------------------
    # Evaluate directly with Keras
    # --------------------------------------------------------

    print(
        "\nRunning Keras evaluation..."
    )

    test_loss, test_accuracy = (
        model.evaluate(
            test_dataset,
            verbose=1,
        )
    )

    print(
        "\nTest loss: "
        f"{test_loss:.4f}"
    )

    print(
        "Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    (
        y_true,
        y_pred,
        y_probability,
    ) = get_predictions(
        model,
        test_dataset,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "\nGenerating confusion matrix..."
    )

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        len(class_names),
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    classification_metrics = (
        print_classification_report(
            class_names,
            calculate_classification_metrics(
                matrix
            ),
        )
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_matrix_path = (
        save_confusion_matrix(
            matrix,
            class_names,
        )
    )

    print(
        f"\nConfusion matrix saved to:"
        f"\n{confusion_matrix_path}"
    )

    # --------------------------------------------------------
    # Save evaluation summary
    # --------------------------------------------------------

    summary_path = (
        save_evaluation_summary(
            test_loss,
            test_accuracy,
            classification_metrics,
        )
    )

    print(
        f"\nEvaluation summary saved to:"
        f"\n{summary_path}"
    )

    # --------------------------------------------------------
    # Most confident wrong predictions
    # --------------------------------------------------------

    wrong_indices = np.where(
        y_true != y_pred
    )[0]

    print("\n")
    print("=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)

    print(
        f"\nIncorrect predictions: "
        f"{len(wrong_indices):,}"
    )

    print(
        f"Correct predictions: "
        f"{len(y_true) - len(wrong_indices):,}"
    )

    print(
        f"Total test images: "
        f"{len(y_true):,}"
    )

    # --------------------------------------------------------
    # Error rate
    # --------------------------------------------------------

    error_rate = (
        len(wrong_indices)
        / len(y_true)
        if len(y_true) > 0
        else 0.0
    )

    print(
        f"Error rate: "
        f"{error_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Top confident mistakes
    # --------------------------------------------------------

    if len(wrong_indices) > 0:

        wrong_confidences = (
            np.max(
                y_probability[
                    wrong_indices
                ],
                axis=1,
            )
        )

        sorted_positions = np.argsort(
            wrong_confidences
        )[::-1]

        print(
            "\nTop 10 confident mistakes:"
        )

        print("-" * 70)

        for position in sorted_positions[:10]:

            index = wrong_indices[
                position
            ]

            confidence = (
                wrong_confidences[
                    position
                ]
            )

            true_class = (
                class_names[
                    y_true[index]
                ]
            )

            predicted_class = (
                class_names[
                    y_pred[index]
                ]
            )

            file_path = (
                test_dataset.file_paths[
                    index
                ]
            )

            print(
                f"\nTrue:      "
                f"{true_class}"
            )

            print(
                f"Predicted: "
                f"{predicted_class}"
            )

            print(
                f"Confidence: "
                f"{confidence:.4f}"
            )

            print(
                f"File:      "
                f"{file_path}"
            )

    else:

        print(
            "\nNo incorrect predictions."
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PHASE 2 TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Final macro F1: "
        f"{classification_metrics['macro_f1']:.4f}"
    )

    print(
        f"Final weighted F1: "
        f"{classification_metrics['weighted_f1']:.4f}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()