from pathlib import Path
import json

import pandas as pd


def load_json(path: Path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_dashboard_data(root: Path):

    robustness_dir = root / "reports" / "robustness"

    results = robustness_dir / "robustness_results.csv"
    per_class = robustness_dir / "per_class_robustness.csv"
    confusion_pairs = robustness_dir / "confusion_pairs.csv"

    clean_report = robustness_dir / "clean_classification_report.json"
    robustness_summary = robustness_dir / "robustness_summary.json"
    gradcam_metadata = (
        root
        / "reports"
        / "gradcam"
        / "gradcam_metadata.json"
    )

    data = {}

    data["robustness_results"] = (
        pd.read_csv(results)
        if results.exists()
        else pd.DataFrame()
    )

    data["per_class_robustness"] = (
        pd.read_csv(per_class)
        if per_class.exists()
        else pd.DataFrame()
    )

    data["confusion_pairs"] = (
        pd.read_csv(confusion_pairs)
        if confusion_pairs.exists()
        else pd.DataFrame()
    )

    data["clean_report"] = load_json(clean_report)
    data["robustness_summary"] = load_json(
        robustness_summary
    )
    data["gradcam_metadata"] = load_json(
        gradcam_metadata
    )

    class_weights = load_json(
        root / "data" / "class_weights.json"
    )

    data["class_weights"] = class_weights

    if class_weights:
        data["num_classes"] = len(class_weights)
    elif data["per_class_robustness"].empty:
        data["num_classes"] = 12
    else:
        data["num_classes"] = len(
            data["per_class_robustness"]
        )

    return data