from pathlib import Path
import json


# ============================================================
# PHASE 3 — EXPLAINABILITY REPORT
# ============================================================

OUTPUT_DIR = Path("models/evaluation/phase3")
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Quantitative evaluation results
# ------------------------------------------------------------

evaluation = {
    "test_images": 2328,
    "correct_predictions": 2236,
    "incorrect_predictions": 92,
    "accuracy": 0.9605,
    "error_rate": 0.0395,
    "macro_f1": 0.9425,
    "weighted_f1": 0.9606,
}


# ------------------------------------------------------------
# Class-level performance
# ------------------------------------------------------------

class_metrics = {
    "battery": {
        "precision": 0.9685,
        "recall": 0.9762,
        "f1": 0.9723,
        "support": 126,
    },
    "biological": {
        "precision": 0.9713,
        "recall": 0.9941,
        "f1": 0.9826,
        "support": 170,
    },
    "brown-glass": {
        "precision": 0.9432,
        "recall": 0.9651,
        "f1": 0.9540,
        "support": 86,
    },
    "cardboard": {
        "precision": 0.9545,
        "recall": 0.9459,
        "f1": 0.9502,
        "support": 111,
    },
    "clothes": {
        "precision": 0.9901,
        "recall": 0.9901,
        "f1": 0.9901,
        "support": 809,
    },
    "green-glass": {
        "precision": 0.9802,
        "recall": 0.9252,
        "f1": 0.9519,
        "support": 107,
    },
    "metal": {
        "precision": 0.9060,
        "recall": 0.9217,
        "f1": 0.9138,
        "support": 115,
    },
    "paper": {
        "precision": 0.9551,
        "recall": 0.9189,
        "f1": 0.9366,
        "support": 185,
    },
    "plastic": {
        "precision": 0.8571,
        "recall": 0.8837,
        "f1": 0.8702,
        "support": 129,
    },
    "shoes": {
        "precision": 0.9726,
        "recall": 0.9827,
        "f1": 0.9776,
        "support": 289,
    },
    "trash": {
        "precision": 0.9905,
        "recall": 0.9369,
        "f1": 0.9630,
        "support": 111,
    },
    "white-glass": {
        "precision": 0.8298,
        "recall": 0.8667,
        "f1": 0.8478,
        "support": 90,
    },
}


# ------------------------------------------------------------
# Important confusion pairs
# ------------------------------------------------------------

confusion_pairs = [
    ("white-glass", "plastic", 8),
    ("clothes", "shoes", 7),
    ("cardboard", "paper", 5),
    ("metal", "white-glass", 5),
    ("plastic", "white-glass", 5),
    ("paper", "cardboard", 4),
    ("shoes", "clothes", 4),
    ("green-glass", "white-glass", 3),
    ("green-glass", "plastic", 3),
    ("metal", "plastic", 3),
    ("paper", "battery", 3),
    ("paper", "biological", 3),
    ("plastic", "metal", 3),
    ("plastic", "paper", 3),
    ("white-glass", "metal", 3),
]


# ------------------------------------------------------------
# Grad-CAM examples
# ------------------------------------------------------------

gradcam_examples = [
    {
        "file": "01_shoes_to_clothes.png",
        "true_class": "shoes",
        "predicted_class": "clothes",
        "confidence": 1.0000,
    },
    {
        "file": "02_green-glass_to_brown-glass.png",
        "true_class": "green-glass",
        "predicted_class": "brown-glass",
        "confidence": 0.9999,
    },
    {
        "file": "03_clothes_to_shoes.png",
        "true_class": "clothes",
        "predicted_class": "shoes",
        "confidence": 0.9991,
    },
    {
        "file": "04_white-glass_to_plastic.png",
        "true_class": "white-glass",
        "predicted_class": "plastic",
        "confidence": 0.9943,
    },
    {
        "file": "05_paper_to_battery.png",
        "true_class": "paper",
        "predicted_class": "battery",
        "confidence": 0.9940,
    },
    {
        "file": "06_cardboard_to_paper.png",
        "true_class": "cardboard",
        "predicted_class": "paper",
        "confidence": 0.9887,
    },
    {
        "file": "07_plastic_to_clothes.png",
        "true_class": "plastic",
        "predicted_class": "clothes",
        "confidence": 0.9863,
    },
    {
        "file": "08_trash_to_white-glass.png",
        "true_class": "trash",
        "predicted_class": "white-glass",
        "confidence": 0.9936,
    },
]


# ------------------------------------------------------------
# Generate text report
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "phase3_explainability_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 72 + "\n")
    f.write("PHASE 3 — MODEL EXPLAINABILITY REPORT\n")
    f.write("=" * 72 + "\n\n")

    f.write("MODEL\n")
    f.write("-" * 72 + "\n")
    f.write("Model: waste_classifier_finetuned.keras\n")
    f.write("Architecture: EfficientNetB0-based image classifier\n")
    f.write("Input size: 224 x 224 x 3\n")
    f.write("Number of classes: 12\n\n")

    f.write("OVERALL PERFORMANCE\n")
    f.write("-" * 72 + "\n")
    f.write(f"Test images:          {evaluation['test_images']:,}\n")
    f.write(f"Correct predictions:  {evaluation['correct_predictions']:,}\n")
    f.write(f"Incorrect predictions:{evaluation['incorrect_predictions']:>6,}\n")
    f.write(f"Accuracy:             {evaluation['accuracy']:.2%}\n")
    f.write(f"Error rate:           {evaluation['error_rate']:.2%}\n")
    f.write(f"Macro F1:             {evaluation['macro_f1']:.4f}\n")
    f.write(f"Weighted F1:          {evaluation['weighted_f1']:.4f}\n\n")

    f.write("CLASS-LEVEL PERFORMANCE\n")
    f.write("-" * 72 + "\n")
    f.write(
        f"{'Class':<16}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}\n"
    )

    for name, metrics in class_metrics.items():
        f.write(
            f"{name:<16}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
            f"{metrics['support']:>12}\n"
        )

    f.write("\n")

    f.write("KEY PERFORMANCE OBSERVATIONS\n")
    f.write("-" * 72 + "\n")

    f.write(
        "1. The model achieved 96.05% accuracy on the untouched test set.\n"
    )

    f.write(
        "2. The macro F1-score of 0.9425 indicates strong performance "
        "across the 12 classes while giving each class equal weight.\n"
    )

    f.write(
        "3. White-glass produced the lowest F1-score at 0.8478 and "
        "the lowest precision at 0.8298.\n"
    )

    f.write(
        "4. Plastic produced an F1-score of 0.8702 and showed several "
        "confusions with white-glass, metal, paper, and clothes.\n"
    )

    f.write(
        "5. Clothes achieved an F1-score of 0.9901 and represented "
        "the largest test class with 809 images.\n"
    )

    f.write(
        "6. Biological achieved the highest recall at 0.9941.\n"
    )

    f.write("\n")

    f.write("TOP CONFUSION PAIRS\n")
    f.write("-" * 72 + "\n")
    f.write(
        f"{'True Class':<18}"
        f"{'Predicted Class':<20}"
        f"{'Count':>8}\n"
    )

    for true_class, predicted_class, count in confusion_pairs:
        f.write(
            f"{true_class:<18}"
            f"{predicted_class:<20}"
            f"{count:>8}\n"
        )

    f.write("\n")

    f.write("GRAD-CAM ANALYSIS\n")
    f.write("-" * 72 + "\n")

    f.write(
        "Grad-CAM was applied to representative high-confidence "
        "misclassifications identified during error analysis.\n\n"
    )

    for i, item in enumerate(gradcam_examples, start=1):
        f.write(
            f"{i}. {item['file']}\n"
            f"   True class:       {item['true_class']}\n"
            f"   Predicted class:  {item['predicted_class']}\n"
            f"   Confidence:       {item['confidence']:.4f}\n\n"
        )

    f.write("INTERPRETATION\n")
    f.write("-" * 72 + "\n")

    f.write(
        "The error analysis shows that the remaining model errors are "
        "concentrated in visually similar waste categories. The most "
        "frequent confusions involve white-glass versus plastic, "
        "clothes versus shoes, and cardboard versus paper.\n\n"
    )

    f.write(
        "Several mistakes occur with very high confidence. This indicates "
        "that some visually ambiguous samples are not simply low-confidence "
        "predictions; the model can confidently assign an incorrect class "
        "when visual characteristics overlap between categories.\n\n"
    )

    f.write(
        "Grad-CAM provides a visual explanation of the regions contributing "
        "to these predictions and can therefore be used to investigate "
        "whether the classifier is relying on relevant object regions or "
        "potentially misleading image regions.\n\n"
    )

    f.write("PHASE 3 CONCLUSION\n")
    f.write("-" * 72 + "\n")

    f.write(
        "Phase 3 establishes both quantitative and visual evidence for "
        "model behavior. The classifier performs strongly overall, while "
        "the remaining errors provide specific targets for future model "
        "improvement and robustness testing.\n"
    )


# ------------------------------------------------------------
# JSON summary
# ------------------------------------------------------------

summary = {
    "phase": 3,
    "purpose": "Model explainability and error analysis",
    "model": "models/waste_classifier_finetuned.keras",
    "test_accuracy": evaluation["accuracy"],
    "error_rate": evaluation["error_rate"],
    "macro_f1": evaluation["macro_f1"],
    "weighted_f1": evaluation["weighted_f1"],
    "incorrect_predictions": evaluation["incorrect_predictions"],
    "gradcam_visualizations": len(gradcam_examples),
    "gradcam_directory": str(GRADCAM_DIR),
    "report": str(report_path),
}

json_path = OUTPUT_DIR / "phase3_summary.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


print("=" * 72)
print("PHASE 3 REPORT COMPLETE")
print("=" * 72)
print()
print(f"Report:  {report_path}")
print(f"Summary: {json_path}")
print(f"Grad-CAM images: {len(gradcam_examples)}")
print()