import json
import io
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/waste_classifier_finetuned.keras"

TEST_DIR = "data/splits/test"

REPORT_DIR = Path("reports/robustness")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)

SEVERITIES = [1, 2, 3]

BATCH_SIZE = 64

SEED = 42

np.random.seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMART WASTE CLASSIFIER - ROBUSTNESS EVALUATION")
print("=" * 70)


# ============================================================
# TENSORFLOW / GPU
# ============================================================

print("\nTensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except Exception:
            pass

else:
    print("GPU: NOT DETECTED")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)


# ============================================================
# DISCOVER CLASSES
# ============================================================

test_path = Path(TEST_DIR)

if not test_path.exists():
    raise RuntimeError(
        f"Test directory does not exist: {TEST_DIR}"
    )


class_names = sorted(
    [
        directory.name
        for directory in test_path.iterdir()
        if directory.is_dir()
    ]
)


if not class_names:
    raise RuntimeError(
        f"No class folders found in: {TEST_DIR}"
    )


class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


print("\nClasses:")

for index, name in enumerate(class_names):
    print(f"{index:2d} -> {name}")


# ============================================================
# LOAD TEST IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


samples = []


for class_name in class_names:

    class_dir = test_path / class_name

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() not in image_extensions:
            continue

        samples.append(
            {
                "path": str(image_path),
                "label": class_to_index[class_name],
                "class_name": class_name,
            }
        )


if not samples:
    raise RuntimeError(
        f"No images found in {TEST_DIR}"
    )


print(
    f"\nTotal test images: {len(samples):,}"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    """
    Load an image using the same pixel scale expected
    by the trained model.

    IMPORTANT:
    The model was trained/evaluated using raw pixel values
    in the 0-255 range.

    Therefore we DO NOT divide by 255 here.
    """

    image = Image.open(path).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    return image


# ============================================================
# ROBUSTNESS TRANSFORMATIONS
# ============================================================

def apply_perturbation(
    image,
    condition,
    severity,
):
    """
    Apply an image perturbation.

    Input:
        image -> float32 array in 0-255 range

    Output:
        float32 array in 0-255 range
    """

    # --------------------------------------------------------
    # Convert to uint8 for PIL operations
    # --------------------------------------------------------

    image_uint8 = np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)

    pil = Image.fromarray(
        image_uint8,
        mode="RGB",
    )


    # ========================================================
    # BRIGHTNESS
    # ========================================================

    if condition == "brightness":

        factors = {
            1: 0.75,
            2: 0.50,
            3: 0.30,
        }

        transformed = ImageEnhance.Brightness(
            pil
        ).enhance(
            factors[severity]
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


    # ========================================================
    # CONTRAST
    # ========================================================

    if condition == "contrast":

        factors = {
            1: 0.75,
            2: 0.50,
            3: 0.30,
        }

        transformed = ImageEnhance.Contrast(
            pil
        ).enhance(
            factors[severity]
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


    # ========================================================
    # GAUSSIAN BLUR
    # ========================================================

    if condition == "blur":

        radius = {
            1: 1.0,
            2: 2.0,
            3: 3.5,
        }

        transformed = pil.filter(
            ImageFilter.GaussianBlur(
                radius[severity]
            )
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


    # ========================================================
    # GAUSSIAN NOISE
    # ========================================================

    if condition == "gaussian_noise":

        # Pixel-scale standard deviation.
        #
        # Previous version used values such as 0.03,
        # which only make sense for normalized 0-1 images.
        #
        # The model uses 0-255 inputs.

        sigma = {
            1: 8.0,
            2: 18.0,
            3: 30.0,
        }

        noise = np.random.normal(
            loc=0.0,
            scale=sigma[severity],
            size=image.shape,
        )

        noisy = image + noise

        return np.clip(
            noisy,
            0,
            255,
        ).astype(np.float32)


    # ========================================================
    # ROTATION
    # ========================================================

    if condition == "rotation":

        angles = {
            1: 5,
            2: 10,
            3: 20,
        }

        transformed = pil.rotate(
            angles[severity],
            resample=Image.Resampling.BILINEAR,
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


    # ========================================================
    # HORIZONTAL FLIP
    # ========================================================

    if condition == "horizontal_flip":

        # Horizontal flipping is binary, so the severity
        # levels are intentionally identical.

        transformed = pil.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


    # ========================================================
    # DOWNSCALE + UPSCALE
    # ========================================================

    if condition == "downscale":

        sizes = {
            1: 160,
            2: 112,
            3: 64,
        }

        small = pil.resize(
            (
                sizes[severity],
                sizes[severity],
            ),
            Image.Resampling.BILINEAR,
        )

        restored = small.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )

        return np.asarray(
            restored,
            dtype=np.float32,
        )


    # ========================================================
    # JPEG COMPRESSION
    # ========================================================

    if condition == "jpeg":

        qualities = {
            1: 70,
            2: 40,
            3: 15,
        }

        buffer = io.BytesIO()

        pil.save(
            buffer,
            format="JPEG",
            quality=qualities[severity],
        )

        buffer.seek(0)

        compressed = Image.open(
            buffer
        ).convert("RGB")

        compressed = compressed.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            compressed,
            dtype=np.float32,
        )


    raise ValueError(
        f"Unknown condition: {condition}"
    )


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_images(images):
    """
    Run model inference in batches.

    Images must be in 0-255 pixel scale.
    """

    predicted_labels = []
    confidences = []

    for start_index in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        batch = np.stack(
            images[
                start_index:
                start_index + BATCH_SIZE
            ]
        )

        probabilities = model.predict(
            batch,
            verbose=0,
        )

        predicted_labels.extend(
            np.argmax(
                probabilities,
                axis=1,
            )
        )

        confidences.extend(
            np.max(
                probabilities,
                axis=1,
            )
        )

    return (
        np.asarray(
            predicted_labels,
            dtype=np.int64,
        ),
        np.asarray(
            confidences,
            dtype=np.float32,
        ),
    )


# ============================================================
# LOAD ALL CLEAN IMAGES
# ============================================================

print("\nLoading test images...")

clean_images = []
true_labels = []


load_start = time.perf_counter()


for sample in samples:

    clean_images.append(
        load_image(
            sample["path"]
        )
    )

    true_labels.append(
        sample["label"]
    )


true_labels = np.asarray(
    true_labels,
    dtype=np.int64,
)


load_time = (
    time.perf_counter()
    - load_start
)


print(
    f"Images loaded in "
    f"{load_time:.2f}s"
)


# ============================================================
# INPUT SCALE SANITY CHECK
# ============================================================

print("\nInput scale sanity check:")

all_min = min(
    float(image.min())
    for image in clean_images
)

all_max = max(
    float(image.max())
    for image in clean_images
)

print(
    f"Minimum pixel value: {all_min:.2f}"
)

print(
    f"Maximum pixel value: {all_max:.2f}"
)

if all_max <= 1.0:

    raise RuntimeError(
        "Images appear to be normalized to 0-1. "
        "The robustness pipeline expects 0-255."
    )


print(
    "Input scale verified: 0-255"
)


# ============================================================
# CLEAN TEST SET
# ============================================================

print("\n" + "=" * 70)
print("CLEAN TEST SET")
print("=" * 70)


start = time.perf_counter()


clean_predictions, clean_confidences = (
    predict_images(
        clean_images
    )
)


clean_accuracy = accuracy_score(
    true_labels,
    clean_predictions,
)


clean_f1 = f1_score(
    true_labels,
    clean_predictions,
    average="macro",
)


clean_time = (
    time.perf_counter()
    - start
)


print(
    f"\nClean Accuracy : "
    f"{clean_accuracy:.4f}"
)

print(
    f"Clean Macro F1 : "
    f"{clean_f1:.4f}"
)

print(
    f"Evaluation time: "
    f"{clean_time:.2f}s"
)

print(
    f"Mean confidence: "
    f"{clean_confidences.mean():.4f}"
)


# ============================================================
# CLEAN CLASSIFICATION REPORT
# ============================================================

clean_report = classification_report(
    true_labels,
    clean_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)


with open(
    REPORT_DIR /
    "clean_classification_report.json",
    "w",
) as f:

    json.dump(
        clean_report,
        f,
        indent=2,
    )


# ============================================================
# CLEAN PREDICTION DISTRIBUTION
# ============================================================

prediction_counts = np.bincount(
    clean_predictions,
    minlength=len(class_names),
)


print("\nClean prediction distribution:")

for index, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:15s}: "
        f"{prediction_counts[index]:5d}"
    )


# ============================================================
# ROBUSTNESS CONDITIONS
# ============================================================

conditions = [
    "brightness",
    "contrast",
    "gaussian_noise",
    "blur",
    "rotation",
    "horizontal_flip",
    "downscale",
    "jpeg",
]


results = []


# ============================================================
# ROBUSTNESS EVALUATION
# ============================================================

for condition in conditions:

    print("\n" + "=" * 70)

    print(
        f"TESTING: "
        f"{condition.upper()}"
    )

    print("=" * 70)


    for severity in SEVERITIES:

        print(
            f"\nSeverity {severity}..."
        )


        # ----------------------------------------------------
        # Apply perturbation
        # ----------------------------------------------------

        transform_start = time.perf_counter()

        transformed_images = []

        for image in clean_images:

            transformed_images.append(
                apply_perturbation(
                    image,
                    condition,
                    severity,
                )
            )


        transform_time = (
            time.perf_counter()
            - transform_start
        )


        # ----------------------------------------------------
        # Verify transformed image scale
        # ----------------------------------------------------

        transformed_max = max(
            float(image.max())
            for image in transformed_images
        )

        transformed_min = min(
            float(image.min())
            for image in transformed_images
        )


        if transformed_max <= 1.0:

            raise RuntimeError(
                f"{condition} severity "
                f"{severity} produced "
                "normalized 0-1 images. "
                "Expected 0-255."
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        start = time.perf_counter()


        predictions, confidence_values = (
            predict_images(
                transformed_images
            )
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            true_labels,
            predictions,
        )


        macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
        )


        accuracy_drop = (
            clean_accuracy
            - accuracy
        )


        f1_drop = (
            clean_f1
            - macro_f1
        )


        relative_accuracy_drop = (
            accuracy_drop
            / clean_accuracy
            if clean_accuracy > 0
            else 0.0
        )


        relative_f1_drop = (
            f1_drop
            / clean_f1
            if clean_f1 > 0
            else 0.0
        )


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Accuracy      : "
            f"{accuracy:.4f}"
        )

        print(
            f"Macro F1      : "
            f"{macro_f1:.4f}"
        )

        print(
            f"Accuracy drop : "
            f"{accuracy_drop:.4f}"
        )

        print(
            f"F1 drop       : "
            f"{f1_drop:.4f}"
        )

        print(
            f"Relative acc. : "
            f"{relative_accuracy_drop:.2%}"
        )

        print(
            f"Relative F1   : "
            f"{relative_f1_drop:.2%}"
        )

        print(
            f"Confidence    : "
            f"{confidence_values.mean():.4f}"
        )

        print(
            f"Pixel range   : "
            f"{transformed_min:.1f} - "
            f"{transformed_max:.1f}"
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "condition": condition,
                "severity": severity,
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "accuracy_drop": float(
                    accuracy_drop
                ),
                "f1_drop": float(
                    f1_drop
                ),
                "relative_accuracy_drop": float(
                    relative_accuracy_drop
                ),
                "relative_f1_drop": float(
                    relative_f1_drop
                ),
                "mean_confidence": float(
                    confidence_values.mean()
                ),
                "transformation_time_seconds": float(
                    transform_time
                ),
                "evaluation_time_seconds": float(
                    elapsed
                ),
                "pixel_min": float(
                    transformed_min
                ),
                "pixel_max": float(
                    transformed_max
                ),
            }
        )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


results_path = (
    REPORT_DIR /
    "robustness_results.csv"
)


results_df.to_csv(
    results_path,
    index=False,
)


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

worst_accuracy = (
    float(results_df["accuracy"].min())
)

worst_f1 = (
    float(results_df["macro_f1"].min())
)

largest_accuracy_drop = (
    float(results_df["accuracy_drop"].max())
)

largest_f1_drop = (
    float(results_df["f1_drop"].max())
)


worst_accuracy_row = (
    results_df.loc[
        results_df["accuracy"].idxmin()
    ]
)


worst_f1_row = (
    results_df.loc[
        results_df["macro_f1"].idxmin()
    ]
)


summary = {
    "model": MODEL_PATH,
    "test_directory": TEST_DIR,
    "test_images": len(samples),
    "image_size": list(IMAGE_SIZE),
    "input_scale": "0-255",
    "batch_size": BATCH_SIZE,
    "classes": class_names,
    "clean_accuracy": float(
        clean_accuracy
    ),
    "clean_macro_f1": float(
        clean_f1
    ),
    "clean_mean_confidence": float(
        clean_confidences.mean()
    ),
    "worst_accuracy": worst_accuracy,
    "worst_macro_f1": worst_f1,
    "largest_accuracy_drop": largest_accuracy_drop,
    "largest_f1_drop": largest_f1_drop,
    "worst_accuracy_condition": (
        str(
            worst_accuracy_row[
                "condition"
            ]
        )
    ),
    "worst_accuracy_severity": int(
        worst_accuracy_row[
            "severity"
        ]
    ),
    "worst_f1_condition": (
        str(
            worst_f1_row[
                "condition"
            ]
        )
    ),
    "worst_f1_severity": int(
        worst_f1_row[
            "severity"
        ]
    ),
    "conditions": conditions,
    "severities": SEVERITIES,
}


summary_path = (
    REPORT_DIR /
    "robustness_summary.json"
)


with open(
    summary_path,
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS EVALUATION COMPLETE")
print("=" * 70)


print(
    f"\nClean Accuracy      : "
    f"{clean_accuracy:.4f}"
)

print(
    f"Clean Macro F1      : "
    f"{clean_f1:.4f}"
)

print(
    f"Worst Accuracy      : "
    f"{worst_accuracy:.4f}"
)

print(
    f"Worst Macro F1      : "
    f"{worst_f1:.4f}"
)

print(
    f"Largest Accuracy Drop: "
    f"{largest_accuracy_drop:.4f}"
)

print(
    f"Largest F1 Drop      : "
    f"{largest_f1_drop:.4f}"
)


print(
    "\nWorst Accuracy Condition:"
)

print(
    f"  {worst_accuracy_row['condition']} "
    f"(severity "
    f"{int(worst_accuracy_row['severity'])})"
)


print(
    "\nWorst F1 Condition:"
)

print(
    f"  {worst_f1_row['condition']} "
    f"(severity "
    f"{int(worst_f1_row['severity'])})"
)


print("\nResults saved to:")

print(
    f"  {results_path}"
)

print(
    f"  {summary_path}"
)

print(
    f"  {REPORT_DIR / 'clean_classification_report.json'}"
)


# ============================================================
# FULL RESULTS
# ============================================================

print("\nResults:")

print(
    results_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("ROBUSTNESS PIPELINE FINISHED")
print("=" * 70)