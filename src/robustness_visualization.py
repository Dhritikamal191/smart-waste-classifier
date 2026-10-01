import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

REPORT_DIR = Path("reports/robustness")
PLOT_DIR = REPORT_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_FILE = (
    REPORT_DIR /
    "robustness_results.csv"
)

PER_CLASS_FILE = (
    REPORT_DIR /
    "per_class_robustness.csv"
)

CONFUSION_FILE = (
    REPORT_DIR /
    "confusion_pairs.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "robustness_error_summary.json"
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.autolayout": True,
    }
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMART WASTE CLASSIFIER - ROBUSTNESS VISUALIZATION")
print("=" * 70)


if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Missing file: {RESULTS_FILE}"
    )


results_df = pd.read_csv(
    RESULTS_FILE
)


print(
    f"\nLoaded robustness results: "
    f"{len(results_df)} rows"
)


per_class_df = None

if PER_CLASS_FILE.exists():

    per_class_df = pd.read_csv(
        PER_CLASS_FILE
    )

    print(
        f"Loaded per-class results: "
        f"{len(per_class_df)} rows"
    )


confusion_df = None

if CONFUSION_FILE.exists():

    confusion_df = pd.read_csv(
        CONFUSION_FILE
    )

    print(
        f"Loaded confusion pairs: "
        f"{len(confusion_df)} rows"
    )


summary = {}

if SUMMARY_FILE.exists():

    with open(
        SUMMARY_FILE,
        "r"
    ) as file:

        summary = json.load(file)


# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """

    path = PLOT_DIR / filename

    plt.savefig(
        path,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def format_condition(value):
    return str(value).replace(
        "_",
        " "
    ).title()


# ============================================================
# 1. ACCURACY BY SEVERITY
# ============================================================

print("\n[1/6] Accuracy degradation plot...")


plt.figure()

for condition in results_df["condition"].unique():

    subset = results_df[
        results_df["condition"] == condition
    ].sort_values("severity")

    plt.plot(
        subset["severity"],
        subset["accuracy"],
        marker="o",
        label=format_condition(condition)
    )


if "clean_accuracy" in summary:

    plt.axhline(
        summary["clean_accuracy"],
        linestyle="--",
        label="Clean Accuracy"
    )


plt.xlabel(
    "Severity"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Model Accuracy Under Image Perturbations"
)

plt.xticks(
    [1, 2, 3]
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.25
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_plot(
    "accuracy_by_severity.png"
)


# ============================================================
# 2. MACRO F1 BY SEVERITY
# ============================================================

print("\n[2/6] Macro F1 degradation plot...")


plt.figure()

for condition in results_df["condition"].unique():

    subset = results_df[
        results_df["condition"] == condition
    ].sort_values("severity")

    plt.plot(
        subset["severity"],
        subset["macro_f1"],
        marker="o",
        label=format_condition(condition)
    )


if "clean_macro_f1" in summary:

    plt.axhline(
        summary["clean_macro_f1"],
        linestyle="--",
        label="Clean Macro F1"
    )


plt.xlabel(
    "Severity"
)

plt.ylabel(
    "Macro F1"
)

plt.title(
    "Macro F1 Under Image Perturbations"
)

plt.xticks(
    [1, 2, 3]
)

plt.ylim(
    0,
    1
)

plt.grid(
    alpha=0.25
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_plot(
    "macro_f1_by_severity.png"
)


# ============================================================
# 3. ACCURACY DROP AT EACH SEVERITY
# ============================================================

print("\n[3/6] Accuracy-drop comparison...")


pivot_accuracy = results_df.pivot(
    index="condition",
    columns="severity",
    values="accuracy_drop"
)


pivot_accuracy = pivot_accuracy.sort_values(
    by=3,
    ascending=False
)


plt.figure(
    figsize=(10, 7)
)

pivot_accuracy.plot(
    kind="bar",
    ax=plt.gca()
)

plt.xlabel(
    "Perturbation"
)

plt.ylabel(
    "Accuracy Drop"
)

plt.title(
    "Accuracy Degradation by Perturbation Severity"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    title="Severity"
)

save_plot(
    "accuracy_drop_by_condition.png"
)


# ============================================================
# 4. WORST CONDITION COMPARISON
# ============================================================

print("\n[4/6] Clean vs worst-condition comparison...")


worst_row = results_df.loc[
    results_df["accuracy"].idxmin()
]


condition = worst_row["condition"]
severity = int(
    worst_row["severity"]
)


clean_accuracy = float(
    summary.get(
        "clean_accuracy",
        results_df["accuracy"].max()
    )
)


worst_accuracy = float(
    worst_row["accuracy"]
)


labels = [
    "Clean",
    (
        f"{format_condition(condition)}\n"
        f"Severity {severity}"
    )
]

values = [
    clean_accuracy,
    worst_accuracy
]


plt.figure()

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Clean Accuracy vs Worst Robustness Condition"
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontweight="bold"
    )


save_plot(
    "clean_vs_worst_condition.png"
)


# ============================================================
# 5. PER-CLASS ROBUSTNESS
# ============================================================

if per_class_df is not None:

    print(
        "\n[5/6] Per-class robustness visualization..."
    )

    # --------------------------------------------------------
    # Detect expected columns
    # --------------------------------------------------------

    print(
        "\nPer-class columns:"
    )

    print(
        per_class_df.columns.tolist()
    )


    # --------------------------------------------------------
    # Find useful columns
    # --------------------------------------------------------

    class_column = None

    for candidate in [
        "class",
        "class_name",
        "true_class"
    ]:

        if candidate in per_class_df.columns:

            class_column = candidate
            break


    if class_column is None:

        print(
            "WARNING: Could not identify class column."
        )

    else:

        # ----------------------------------------------------
        # Identify drop column
        # ----------------------------------------------------

        drop_column = None

        for candidate in [
            "accuracy_drop",
            "drop",
            "accuracy_degradation"
        ]:

            if candidate in per_class_df.columns:

                drop_column = candidate
                break


        if drop_column is not None:

            # ------------------------------------------------
            # Select largest class-level degradation
            # ------------------------------------------------

            severity_column = None

            for candidate in [
                "severity",
                "condition"
            ]:

                if candidate in per_class_df.columns:

                    severity_column = candidate
                    break


            # ------------------------------------------------
            # Pivot when condition exists
            # ------------------------------------------------

            if (
                "condition" in
                per_class_df.columns
            ):

                heatmap_data = (
                    per_class_df
                    .pivot_table(
                        index=class_column,
                        columns="condition",
                        values=drop_column,
                        aggfunc="max"
                    )
                )

                heatmap_data = heatmap_data.fillna(0)

                plt.figure(
                    figsize=(12, 8)
                )

                plt.imshow(
                    heatmap_data.values,
                    aspect="auto"
                )

                plt.colorbar(
                    label="Accuracy Drop"
                )

                plt.xticks(
                    range(
                        len(
                            heatmap_data.columns
                        )
                    ),
                    [
                        format_condition(
                            value
                        )
                        for value in
                        heatmap_data.columns
                    ],
                    rotation=35,
                    ha="right"
                )

                plt.yticks(
                    range(
                        len(
                            heatmap_data.index
                        )
                    ),
                    heatmap_data.index
                )

                plt.xlabel(
                    "Perturbation"
                )

                plt.ylabel(
                    "Class"
                )

                plt.title(
                    "Per-Class Robustness Degradation"
                )

                save_plot(
                    "per_class_robustness_heatmap.png"
                )


# ============================================================
# 6. TOP CONFUSION PAIRS
# ============================================================

if confusion_df is not None:

    print(
        "\n[6/6] Confusion-pair visualization..."
    )


    print(
        "\nConfusion columns:"
    )

    print(
        confusion_df.columns.tolist()
    )


    if "count" in confusion_df.columns:

        confusion_plot = (
            confusion_df
            .sort_values(
                "count",
                ascending=False
            )
            .head(15)
            .copy()
        )


        if (
            "true_class" in
            confusion_plot.columns
            and
            "predicted_class" in
            confusion_plot.columns
        ):

            confusion_plot["pair"] = (
                confusion_plot["true_class"]
                .astype(str)
                + " → "
                + confusion_plot["predicted_class"]
                .astype(str)
            )


            confusion_plot = (
                confusion_plot
                .sort_values(
                    "count"
                )
            )


            plt.figure(
                figsize=(10, 7)
            )

            plt.barh(
                confusion_plot["pair"],
                confusion_plot["count"]
            )

            plt.xlabel(
                "Number of Misclassifications"
            )

            plt.ylabel(
                "True → Predicted"
            )

            plt.title(
                "Top Robustness Confusion Pairs"
            )

            plt.grid(
                axis="x",
                alpha=0.25
            )

            save_plot(
                "top_confusion_pairs.png"
            )


# ============================================================
# SAVE VISUALIZATION SUMMARY
# ============================================================

visualization_summary = {
    "plots_directory": str(PLOT_DIR),
    "plots": [
        "accuracy_by_severity.png",
        "macro_f1_by_severity.png",
        "accuracy_drop_by_condition.png",
        "clean_vs_worst_condition.png",
        "per_class_robustness_heatmap.png",
        "top_confusion_pairs.png",
    ],
    "worst_condition": {
        "condition": str(
            worst_row["condition"]
        ),
        "severity": int(
            worst_row["severity"]
        ),
        "accuracy": float(
            worst_row["accuracy"]
        ),
        "accuracy_drop": float(
            worst_row["accuracy_drop"]
        ),
    }
}


summary_output = (
    REPORT_DIR /
    "visualization_summary.json"
)


with open(
    summary_output,
    "w"
) as file:

    json.dump(
        visualization_summary,
        file,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"\nPlots saved to:"
)

print(
    PLOT_DIR
)

print(
    "\nVisualization summary:"
)

print(
    summary_output
)

print(
    "\nGenerated plots:"
)

for filename in visualization_summary["plots"]:

    path = PLOT_DIR / filename

    if path.exists():

        print(
            f"  ✓ {filename}"
        )

print(
    "\n" + "=" * 70
)
