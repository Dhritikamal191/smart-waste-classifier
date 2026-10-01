# ============================================================
# SMART WASTE CLASSIFIER
# ROBUSTNESS ERROR ANALYSIS
# ============================================================

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/waste_classifier_finetuned.keras"
)

TEST_DIR = Path(
    "data/splits/test"
)

REPORT_DIR = Path(
    "reports/robustness"
)

MISCLASSIFIED_DIR = (
    REPORT_DIR / "misclassified"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 64

SEED = 42

np.random.seed(SEED)


# Conditions that were identified as the most
# important from the previous robustness evaluation.

ANALYSIS_CONDITIONS = {
    "blur": 3,
    "gaussian_noise": 3,
    "contrast": 3,
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if MISCLASSIFIED_DIR.exists():

    shutil.rmtree(
        MISCLASSIFIED_DIR
    )

MISCLASSIFIED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "SMART WASTE CLASSIFIER - ROBUSTNESS ERROR ANALYSIS"
)
print("=" * 70)


# ============================================================
# TENSORFLOW / GPU
# ============================================================

print(
    f"\nTensorFlow: {tf.__version__}"
)

gpus = tf.config.list_physical_devices(
    "GPU"
)

if gpus:

    print(
        "GPU:",
        gpus
    )

else:

    print(
        "GPU: NOT DETECTED"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully."
)

print(
    "\nModel input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# DISCOVER CLASSES
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


class_names = sorted(
    [
        directory.name
        for directory in TEST_DIR.iterdir()
        if directory.is_dir()
    ]
)


if not class_names:

    raise RuntimeError(
        f"No class folders found in {TEST_DIR}"
    )


class_to_index = {
    name: index
    for index, name in enumerate(
        class_names
    )
}


print("\nClasses:")

for index, name in enumerate(
    class_names
):

    print(
        f"{index:2d} -> {name}"
    )


# ============================================================
# DISCOVER TEST IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


samples = []


for class_name in class_names:

    class_dir = (
        TEST_DIR / class_name
    )

    for image_path in class_dir.rglob("*"):

        if (
            image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        samples.append(
            {
                "path": image_path,
                "label": class_to_index[
                    class_name
                ],
                "class_name": class_name,
            }
        )


if not samples:

    raise RuntimeError(
        f"No test images found in {TEST_DIR}"
    )


print(
    f"\nTotal test images: {len(samples):,}"
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    """
    Load image in the same 0-255 scale
    used by the trained EfficientNet model.

    IMPORTANT:
    EfficientNetB0 in this project contains
    its own preprocessing behavior, so we
    intentionally DO NOT divide by 255 here.
    """

    image = Image.open(
        path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image


# ============================================================
# PERTURBATION
# ============================================================

def apply_perturbation(
    image,
    condition,
    severity,
):
    """
    Apply a robustness transformation.

    Input:
        image: float32 image in 0-255 range.

    Output:
        float32 image in 0-255 range.
    """

    image_uint8 = np.clip(
        image,
        0,
        255
    ).astype(
        np.uint8
    )

    pil = Image.fromarray(
        image_uint8
    )


    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if condition == "blur":

        radius = {
            1: 1.0,
            2: 2.0,
            3: 3.5,
        }

        result = pil.filter(
            ImageFilter.GaussianBlur(
                radius[
                    severity
                ]
            )
        )

        return np.asarray(
            result,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    if condition == "gaussian_noise":

        sigma = {
            1: 0.03,
            2: 0.07,
            3: 0.12,
        }

        normalized = (
            image / 255.0
        )

        noise = np.random.normal(
            0,
            sigma[
                severity
            ],
            normalized.shape
        )

        noisy = (
            normalized +
            noise
        )

        noisy = np.clip(
            noisy,
            0,
            1
        )

        return (
            noisy * 255.0
        ).astype(
            np.float32
        )


    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    if condition == "contrast":

        factors = {
            1: 0.75,
            2: 0.50,
            3: 0.30,
        }

        result = ImageEnhance.Contrast(
            pil
        ).enhance(
            factors[
                severity
            ]
        )

        return np.asarray(
            result,
            dtype=np.float32
        )


    raise ValueError(
        f"Unknown condition: {condition}"
    )


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_images(
    images
):

    predictions = []

    confidences = []

    for start in range(
        0,
        len(images),
        BATCH_SIZE
    ):

        batch = np.stack(
            images[
                start:
                start + BATCH_SIZE
            ]
        )

        probabilities = model.predict(
            batch,
            verbose=0
        )

        predictions.extend(
            np.argmax(
                probabilities,
                axis=1
            )
        )

        confidences.extend(
            np.max(
                probabilities,
                axis=1
            )
        )

    return (
        np.asarray(
            predictions
        ),
        np.asarray(
            confidences
        ),
    )


# ============================================================
# LOAD ALL TEST IMAGES
# ============================================================

print(
    "\nLoading test images..."
)

images = []

true_labels = []

paths = []

for sample in samples:

    images.append(
        load_image(
            sample["path"]
        )
    )

    true_labels.append(
        sample["label"]
    )

    paths.append(
        sample["path"]
    )


true_labels = np.asarray(
    true_labels
)


print(
    f"Loaded {len(images):,} images."
)


# ============================================================
# INPUT SCALE CHECK
# ============================================================

all_min = min(
    image.min()
    for image in images
)

all_max = max(
    image.max()
    for image in images
)

print(
    "\nInput scale:"
)

print(
    f"Minimum: {all_min:.2f}"
)

print(
    f"Maximum: {all_max:.2f}"
)


if all_max <= 1.0:

    raise RuntimeError(
        "Images appear to be normalized to 0-1. "
        "This analysis expects 0-255 input."
    )


print(
    "Input scale verified: 0-255"
)


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CLEAN TEST SET"
)

print(
    "=" * 70
)


clean_predictions, clean_confidences = (
    predict_images(
        images
    )
)


clean_accuracy = accuracy_score(
    true_labels,
    clean_predictions
)


clean_report = classification_report(
    true_labels,
    clean_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)


print(
    f"\nClean accuracy: "
    f"{clean_accuracy:.4f}"
)


# ============================================================
# CLEAN CONFUSION MATRIX
# ============================================================

clean_cm = confusion_matrix(
    true_labels,
    clean_predictions,
    labels=range(
        len(class_names)
    ),
)


pd.DataFrame(
    clean_cm,
    index=class_names,
    columns=class_names,
).to_csv(
    REPORT_DIR /
    "confusion_matrix_clean.csv"
)


# ============================================================
# CLEAN CLASS METRICS
# ============================================================

clean_class_metrics = []


for index, class_name in enumerate(
    class_names
):

    true_mask = (
        true_labels == index
    )

    class_accuracy = (
        np.mean(
            clean_predictions[
                true_mask
            ] == index
        )
        if true_mask.any()
        else 0.0
    )

    clean_class_metrics.append(
        {
            "class": class_name,
            "clean_accuracy": class_accuracy,
            "precision": clean_report[
                class_name
            ]["precision"],
            "recall": clean_report[
                class_name
            ]["recall"],
            "f1": clean_report[
                class_name
            ]["f1-score"],
            "support": int(
                true_mask.sum()
            ),
        }
    )


# ============================================================
# ANALYZE EACH CONDITION
# ============================================================

per_class_results = []

confusion_pairs = []


for condition, severity in (
    ANALYSIS_CONDITIONS.items()
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"{condition.upper()} "
        f"- SEVERITY {severity}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # TRANSFORM IMAGES
    # --------------------------------------------------------

    transformed_images = []

    for image in images:

        transformed_images.append(
            apply_perturbation(
                image,
                condition,
                severity
            )
        )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions, confidences = (
        predict_images(
            transformed_images
        )
    )


    accuracy = accuracy_score(
        true_labels,
        predictions
    )


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean confidence: "
        f"{confidences.mean():.4f}"
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=range(
            len(class_names)
        ),
    )


    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )


    cm_df.to_csv(
        REPORT_DIR /
        f"confusion_matrix_"
        f"{condition}_severity_"
        f"{severity}.csv"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


    # --------------------------------------------------------
    # PER CLASS METRICS
    # --------------------------------------------------------

    for index, class_name in enumerate(
        class_names
    ):

        mask = (
            true_labels == index
        )

        class_accuracy = (
            np.mean(
                predictions[
                    mask
                ] == index
            )
            if mask.any()
            else 0.0
        )


        clean_accuracy_class = (
            clean_class_metrics[
                index
            ]["clean_accuracy"]
        )


        accuracy_drop = (
            clean_accuracy_class -
            class_accuracy
        )


        per_class_results.append(
            {
                "condition": condition,
                "severity": severity,
                "class": class_name,
                "clean_accuracy":
                    clean_accuracy_class,
                "perturbed_accuracy":
                    class_accuracy,
                "accuracy_drop":
                    accuracy_drop,
                "precision":
                    report[class_name][
                        "precision"
                    ],
                "recall":
                    report[class_name][
                        "recall"
                    ],
                "f1":
                    report[class_name][
                        "f1-score"
                    ],
                "support":
                    int(mask.sum()),
            }
        )


    # --------------------------------------------------------
    # CONFUSION PAIRS
    # --------------------------------------------------------

    for true_index in range(
        len(class_names)
    ):

        for predicted_index in range(
            len(class_names)
        ):

            if (
                true_index ==
                predicted_index
            ):
                continue


            count = cm[
                true_index,
                predicted_index
            ]


            if count > 0:

                confusion_pairs.append(
                    {
                        "condition":
                            condition,
                        "severity":
                            severity,
                        "true_class":
                            class_names[
                                true_index
                            ],
                        "predicted_class":
                            class_names[
                                predicted_index
                            ],
                        "count":
                            int(count),
                    }
                )


    # --------------------------------------------------------
    # SAVE MISCLASSIFIED EXAMPLES
    # --------------------------------------------------------

    condition_dir = (
        MISCLASSIFIED_DIR /
        condition
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    misclassified_indices = np.where(
        predictions != true_labels
    )[0]


    # Limit saved images so the report
    # directory does not become unnecessarily huge.

    MAX_SAVED_IMAGES = 100


    saved_count = 0


    for index in misclassified_indices:

        if (
            saved_count >=
            MAX_SAVED_IMAGES
        ):
            break


        original_path = paths[
            index
        ]


        true_class = class_names[
            true_labels[index]
        ]


        predicted_class = class_names[
            predictions[index]
        ]


        filename = (
            f"{saved_count:03d}_"
            f"true-{true_class}_"
            f"pred-{predicted_class}_"
            f"{original_path.name}"
        )


        destination = (
            condition_dir /
            filename
        )


        # Save the transformed image,
        # because this is what caused the error.

        transformed_uint8 = np.clip(
            transformed_images[
                index
            ],
            0,
            255
        ).astype(
            np.uint8
        )


        Image.fromarray(
            transformed_uint8
        ).save(
            destination
        )


        saved_count += 1


    print(
        f"Misclassified images: "
        f"{len(misclassified_indices):,}"
    )

    print(
        f"Saved examples: "
        f"{saved_count}"
    )


# ============================================================
# SAVE PER-CLASS RESULTS
# ============================================================

per_class_df = pd.DataFrame(
    per_class_results
)


per_class_path = (
    REPORT_DIR /
    "per_class_robustness.csv"
)


per_class_df.to_csv(
    per_class_path,
    index=False
)


# ============================================================
# SAVE CONFUSION PAIRS
# ============================================================

confusion_pairs_df = pd.DataFrame(
    confusion_pairs
)


if not confusion_pairs_df.empty:

    confusion_pairs_df = (
        confusion_pairs_df.sort_values(
            "count",
            ascending=False
        )
    )


confusion_pairs_path = (
    REPORT_DIR /
    "confusion_pairs.csv"
)


confusion_pairs_df.to_csv(
    confusion_pairs_path,
    index=False
)


# ============================================================
# FIND MOST AFFECTED CLASSES
# ============================================================

most_affected = []


for condition in (
    ANALYSIS_CONDITIONS.keys()
):

    subset = per_class_df[
        per_class_df[
            "condition"
        ] == condition
    ]


    subset = subset.sort_values(
        "accuracy_drop",
        ascending=False
    )


    for _, row in (
        subset.head(5).iterrows()
    ):

        most_affected.append(
            {
                "condition":
                    condition,
                "class":
                    row["class"],
                "accuracy_drop":
                    float(
                        row[
                            "accuracy_drop"
                        ]
                    ),
                "perturbed_accuracy":
                    float(
                        row[
                            "perturbed_accuracy"
                        ]
                    ),
            }
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "model": str(
        MODEL_PATH
    ),

    "test_images": int(
        len(samples)
    ),

    "classes": class_names,

    "input_scale": "0-255",

    "clean_accuracy": float(
        clean_accuracy
    ),

    "clean_mean_confidence": float(
        clean_confidences.mean()
    ),

    "analysis_conditions": {
        condition: int(
            severity
        )
        for condition, severity
        in ANALYSIS_CONDITIONS.items()
    },

    "most_affected_classes":
        most_affected,

    "reports": {
        "per_class":
            str(per_class_path),

        "confusion_pairs":
            str(confusion_pairs_path),

        "clean_confusion_matrix":
            str(
                REPORT_DIR /
                "confusion_matrix_clean.csv"
            ),
    },
}


summary_path = (
    REPORT_DIR /
    "robustness_error_summary.json"
)


with open(
    summary_path,
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=2
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ROBUSTNESS ERROR ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nClean accuracy:"
)

print(
    f"{clean_accuracy:.4f}"
)


print(
    "\nMost affected classes:"
)


for item in most_affected:

    print(
        f"{item['condition']:18s} "
        f"{item['class']:15s} "
        f"drop="
        f"{item['accuracy_drop']:.4f}"
    )


print(
    "\nTop confusion pairs:"
)


if confusion_pairs_df.empty:

    print(
        "No confusion pairs found."
    )

else:

    print(
        confusion_pairs_df.head(
            15
        ).to_string(
            index=False
        )
    )


print(
    "\nReports saved to:"
)

print(
    REPORT_DIR
)

print(
    "\nPer-class report:"
)

print(
    per_class_path
)

print(
    "\nConfusion pairs:"
)

print(
    confusion_pairs_path
)

print(
    "\nSummary:"
)

print(
    summary_path
)

print(
    "\nMisclassified examples:"
)

print(
    MISCLASSIFIED_DIR
)

print(
    "\n" + "=" * 70
)

print(
    "PIPELINE FINISHED"
)

print(
    "=" * 70
)