from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score


# ============================================================
# PHASE 4 — ROBUSTNESS TESTING
# ============================================================

MODEL_PATH = Path("models/waste_classifier_finetuned.keras")
TEST_DIR = Path("data/splits/test")
OUTPUT_DIR = Path("models/evaluation/phase4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
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
]


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("=" * 72)
print("PHASE 4 — ROBUSTNESS TESTING")
print("=" * 72)
print()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print()


# ------------------------------------------------------------
# Collect test images
# ------------------------------------------------------------

samples = []

for class_name in CLASS_NAMES:

    class_dir = TEST_DIR / class_name

    if not class_dir.exists():
        continue

    for image_path in sorted(class_dir.glob("*")):

        if image_path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }:
            continue

        samples.append(
            {
                "path": image_path,
                "label": class_name,
                "label_index": CLASS_NAMES.index(class_name),
            }
        )


print(f"Test images found: {len(samples):,}")
print()


# ------------------------------------------------------------
# Image loading
# ------------------------------------------------------------

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)

    return image, array


def preprocess(image):
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    return array


# ------------------------------------------------------------
# Robustness transformations
# ------------------------------------------------------------

def original(image):
    return image.copy()


def brightness(image):
    return ImageEnhance.Brightness(image).enhance(0.65)


def contrast(image):
    return ImageEnhance.Contrast(image).enhance(0.65)


def rotation(image):
    return image.rotate(15, resample=Image.Resampling.BILINEAR)


def horizontal_flip(image):
    return ImageOps.mirror(image)


def blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1.5))


def noise(image):

    array = np.asarray(image, dtype=np.float32)

    noise_array = np.random.normal(
        loc=0.0,
        scale=12.0,
        size=array.shape,
    )

    noisy = np.clip(array + noise_array, 0, 255)

    return Image.fromarray(noisy.astype(np.uint8))


TRANSFORMATIONS = {
    "original": original,
    "brightness": brightness,
    "contrast": contrast,
    "rotation": rotation,
    "horizontal_flip": horizontal_flip,
    "blur": blur,
    "noise": noise,
}


# ------------------------------------------------------------
# Evaluate one transformation
# ------------------------------------------------------------

def evaluate_transformation(name, transform):

    print("-" * 72)
    print(f"Testing: {name}")
    print("-" * 72)

    true_labels = []
    predicted_labels = []
    confidences = []

    for index, sample in enumerate(samples, start=1):

        image, _ = load_image(sample["path"])

        transformed = transform(image)

        inputs = preprocess(transformed)

        probabilities = model.predict(
            inputs,
            verbose=0,
        )[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        true_labels.append(sample["label_index"])
        predicted_labels.append(predicted_index)
        confidences.append(confidence)

        if index % 500 == 0:
            print(
                f"Processed {index:,}/{len(samples):,}"
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    mean_confidence = float(
        np.mean(confidences)
    )

    return {
        "accuracy": float(accuracy),
        "error_rate": float(1.0 - accuracy),
        "mean_confidence": mean_confidence,
        "samples": len(samples),
    }


# ------------------------------------------------------------
# Run robustness tests
# ------------------------------------------------------------

results = {}

for name, transform in TRANSFORMATIONS.items():

    results[name] = evaluate_transformation(
        name,
        transform,
    )


# ------------------------------------------------------------
# Calculate degradation relative to original
# ------------------------------------------------------------

baseline_accuracy = results["original"]["accuracy"]
baseline_confidence = results["original"]["mean_confidence"]

for name in results:

    results[name]["accuracy_drop"] = (
        baseline_accuracy
        - results[name]["accuracy"]
    )

    results[name]["confidence_drop"] = (
        baseline_confidence
        - results[name]["mean_confidence"]
    )


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("=" * 72)
print("ROBUSTNESS RESULTS")
print("=" * 72)

print(
    f"{'Condition':<22}"
    f"{'Accuracy':>12}"
    f"{'Error Rate':>14}"
    f"{'Confidence':>16}"
    f"{'Accuracy Drop':>16}"
)

print("-" * 80)

for name, result in results.items():

    print(
        f"{name:<22}"
        f"{result['accuracy']:>11.2%}"
        f"{result['error_rate']:>13.2%}"
        f"{result['mean_confidence']:>15.4f}"
        f"{result['accuracy_drop']:>15.2%}"
    )


# ------------------------------------------------------------
# Identify strongest and weakest conditions
# ------------------------------------------------------------

non_baseline = {
    name: result
    for name, result in results.items()
    if name != "original"
}

best_condition = max(
    non_baseline,
    key=lambda name: non_baseline[name]["accuracy"],
)

worst_condition = min(
    non_baseline,
    key=lambda name: non_baseline[name]["accuracy"],
)


print()
print("=" * 72)
print("ROBUSTNESS SUMMARY")
print("=" * 72)

print(
    f"Baseline accuracy:        "
    f"{baseline_accuracy:.2%}"
)

print(
    f"Best transformed accuracy:"
    f" {results[best_condition]['accuracy']:.2%}"
    f" ({best_condition})"
)

print(
    f"Worst transformed accuracy:"
    f" {results[worst_condition]['accuracy']:.2%}"
    f" ({worst_condition})"
)

print(
    f"Largest accuracy drop:    "
    f"{results[worst_condition]['accuracy_drop']:.2%}"
    f" ({worst_condition})"
)


# ------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------

json_path = OUTPUT_DIR / "robustness_results.json"

output = {
    "phase": 4,
    "purpose": "Robustness testing",
    "model": str(MODEL_PATH),
    "test_samples": len(samples),
    "baseline_accuracy": baseline_accuracy,
    "baseline_confidence": baseline_confidence,
    "results": results,
    "best_transformed_condition": best_condition,
    "worst_transformed_condition": worst_condition,
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)


# ------------------------------------------------------------
# Save human-readable report
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "robustness_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 72 + "\n")
    f.write("PHASE 4 — ROBUSTNESS TESTING REPORT\n")
    f.write("=" * 72 + "\n\n")

    f.write("MODEL\n")
    f.write("-" * 72 + "\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write("Architecture: EfficientNetB0-based classifier\n")
    f.write("Input size: 224 x 224\n")
    f.write(f"Test samples: {len(samples):,}\n\n")

    f.write("BASELINE\n")
    f.write("-" * 72 + "\n")
    f.write(f"Accuracy: {baseline_accuracy:.4f}\n")
    f.write(f"Accuracy: {baseline_accuracy:.2%}\n")
    f.write(
        f"Mean confidence: {baseline_confidence:.4f}\n\n"
    )

    f.write("ROBUSTNESS RESULTS\n")
    f.write("-" * 72 + "\n")

    for name, result in results.items():

        f.write(
            f"{name}\n"
            f"  Accuracy:          {result['accuracy']:.4f}\n"
            f"  Error rate:        {result['error_rate']:.4f}\n"
            f"  Mean confidence:   {result['mean_confidence']:.4f}\n"
            f"  Accuracy drop:     {result['accuracy_drop']:.4f}\n"
            f"  Confidence drop:   {result['confidence_drop']:.4f}\n\n"
        )

    f.write("INTERPRETATION\n")
    f.write("-" * 72 + "\n")

    f.write(
        "Robustness testing evaluates whether model performance "
        "remains stable when test images are subjected to controlled "
        "visual perturbations.\n\n"
    )

    f.write(
        f"The original test accuracy was {baseline_accuracy:.2%}. "
        f"The weakest tested transformation was "
        f"{worst_condition}, with an accuracy of "
        f"{results[worst_condition]['accuracy']:.2%}.\n\n"
    )

    f.write(
        "Accuracy degradation under image transformations provides "
        "an indication of the model's sensitivity to changes in "
        "image appearance. A small degradation suggests greater "
        "robustness, while a large degradation identifies a "
        "potential area for future improvement.\n\n"
    )

    f.write("CONCLUSION\n")
    f.write("-" * 72 + "\n")

    f.write(
        "Phase 4 evaluates the reliability of the trained classifier "
        "under controlled image perturbations without modifying or "
        "retraining the model.\n"
    )


print()
print("=" * 72)
print("PHASE 4 ROBUSTNESS TESTING COMPLETE")
print("=" * 72)
print()
print(f"Results: {json_path}")
print(f"Report:  {report_path}")