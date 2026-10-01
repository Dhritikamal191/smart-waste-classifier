import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Error Analysis | Smart Waste Classifier",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

ROBUSTNESS_DIR = REPORTS_DIR / "robustness"

ERROR_SUMMARY_PATH = (
    ROBUSTNESS_DIR / "robustness_error_summary.json"
)

CONFUSION_PAIRS_PATH = (
    ROBUSTNESS_DIR / "confusion_pairs.csv"
)

PER_CLASS_PATH = (
    ROBUSTNESS_DIR / "per_class_robustness.csv"
)

CLEAN_REPORT_PATH = (
    ROBUSTNESS_DIR / "clean_classification_report.json"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔍 Error Analysis")

st.caption(
    "Analysis of model failures, confusion patterns, "
    "class-level weaknesses, and robustness-related errors."
)


# ============================================================
# LOAD DATA
# ============================================================

def load_json(path: Path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def load_csv(path: Path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


error_summary = load_json(ERROR_SUMMARY_PATH)
confusion_pairs = load_csv(CONFUSION_PAIRS_PATH)
per_class = load_csv(PER_CLASS_PATH)
clean_report = load_json(CLEAN_REPORT_PATH)


# ============================================================
# DATA AVAILABILITY
# ============================================================

if (
    error_summary is None
    and confusion_pairs is None
    and per_class is None
    and clean_report is None
):

    st.error(
        "No error-analysis reports were found."
    )

    st.info(
        "Expected reports under: "
        f"{ROBUSTNESS_DIR}"
    )

    st.stop()


# ============================================================
# SUMMARY METRICS
# ============================================================

total_errors = 0
total_samples = 0
clean_accuracy = None
worst_class = None
worst_accuracy = None


if isinstance(error_summary, dict):

    total_errors = error_summary.get(
        "total_errors",
        error_summary.get(
            "errors",
            0,
        ),
    )

    total_samples = error_summary.get(
        "total_samples",
        error_summary.get(
            "samples",
            0,
        ),
    )


if isinstance(clean_report, dict):

    clean_accuracy = clean_report.get(
        "accuracy"
    )


if per_class is not None and not per_class.empty:

    possible_class_columns = [
        "class",
        "class_name",
        "label",
    ]

    possible_accuracy_columns = [
        "accuracy",
        "recall",
        "f1",
        "f1_score",
    ]

    class_column = next(
        (
            column
            for column in possible_class_columns
            if column in per_class.columns
        ),
        None,
    )

    accuracy_column = next(
        (
            column
            for column in possible_accuracy_columns
            if column in per_class.columns
        ),
        None,
    )

    if (
        class_column is not None
        and accuracy_column is not None
    ):

        temp = per_class.copy()

        temp[accuracy_column] = pd.to_numeric(
            temp[accuracy_column],
            errors="coerce",
        )

        temp = temp.dropna(
            subset=[accuracy_column]
        )

        if not temp.empty:

            worst_row = temp.loc[
                temp[accuracy_column].idxmin()
            ]

            worst_class = worst_row[
                class_column
            ]

            worst_accuracy = worst_row[
                accuracy_column
            ]


# ============================================================
# KPI ROW
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Samples",
        (
            f"{total_samples:,}"
            if total_samples
            else "—"
        ),
    )


with col2:

    st.metric(
        "Total Errors",
        (
            f"{total_errors:,}"
            if total_errors
            else "—"
        ),
    )


with col3:

    if clean_accuracy is not None:

        st.metric(
            "Clean Accuracy",
            f"{clean_accuracy * 100:.2f}%",
        )

    else:

        st.metric(
            "Clean Accuracy",
            "—",
        )


with col4:

    if (
        worst_class is not None
        and worst_accuracy is not None
    ):

        st.metric(
            "Weakest Class",
            str(worst_class).replace(
                "-",
                " ",
            ).title(),
            f"{worst_accuracy * 100:.2f}%",
        )

    else:

        st.metric(
            "Weakest Class",
            "—",
        )


st.divider()


# ============================================================
# CONFUSION PAIRS
# ============================================================

st.subheader("Most Frequent Confusion Pairs")

if (
    confusion_pairs is not None
    and not confusion_pairs.empty
):

    df = confusion_pairs.copy()

    # Detect columns automatically.

    actual_column = next(
        (
            column
            for column in [
                "actual",
                "true_class",
                "true",
                "actual_class",
            ]
            if column in df.columns
        ),
        None,
    )

    predicted_column = next(
        (
            column
            for column in [
                "predicted",
                "pred_class",
                "prediction",
                "predicted_class",
            ]
            if column in df.columns
        ),
        None,
    )

    count_column = next(
        (
            column
            for column in [
                "count",
                "errors",
                "frequency",
                "confusion_count",
            ]
            if column in df.columns
        ),
        None,
    )

    if (
        actual_column
        and predicted_column
        and count_column
    ):

        df[count_column] = pd.to_numeric(
            df[count_column],
            errors="coerce",
        )

        df = df.dropna(
            subset=[count_column]
        )

        df = df.sort_values(
            count_column,
            ascending=False,
        ).head(15)

        df["actual_label"] = (
            df[actual_column]
            .astype(str)
            .str.replace(
                "-",
                " ",
            )
            .str.title()
        )

        df["predicted_label"] = (
            df[predicted_column]
            .astype(str)
            .str.replace(
                "-",
                " ",
            )
            .str.title()
        )

        df["confusion_pair"] = (
            df["actual_label"]
            + " → "
            + df["predicted_label"]
        )

        fig = px.bar(
            df,
            x=count_column,
            y="confusion_pair",
            orientation="h",
            title="Top Model Confusion Pairs",
            labels={
                count_column: "Number of Errors",
                "confusion_pair": "",
            },
        )

        fig.update_layout(
            height=520,
            yaxis={
                "categoryorder": "total ascending"
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            df[
                [
                    "actual_label",
                    "predicted_label",
                    count_column,
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "Confusion-pair data is not available."
    )


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

st.divider()

st.subheader("Per-Class Performance")

if (
    per_class is not None
    and not per_class.empty
):

    df = per_class.copy()

    class_column = next(
        (
            column
            for column in [
                "class",
                "class_name",
                "label",
            ]
            if column in df.columns
        ),
        None,
    )

    metric_candidates = [
        "accuracy",
        "recall",
        "f1",
        "f1_score",
        "precision",
    ]

    available_metrics = [
        column
        for column in metric_candidates
        if column in df.columns
    ]

    if (
        class_column is not None
        and available_metrics
    ):

        selected_metric = st.selectbox(
            "Performance metric",
            available_metrics,
        )

        plot_df = df.copy()

        plot_df[selected_metric] = pd.to_numeric(
            plot_df[selected_metric],
            errors="coerce",
        )

        plot_df = plot_df.dropna(
            subset=[selected_metric]
        )

        plot_df["display_class"] = (
            plot_df[class_column]
            .astype(str)
            .str.replace(
                "-",
                " ",
            )
            .str.title()
        )

        plot_df = plot_df.sort_values(
            selected_metric
        )

        fig = px.bar(
            plot_df,
            x=selected_metric,
            y="display_class",
            orientation="h",
            title=(
                f"Per-Class {selected_metric.title()}"
            ),
            labels={
                selected_metric: (
                    selected_metric.title()
                ),
                "display_class": "",
            },
        )

        fig.update_layout(
            height=560,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Per-class robustness data is not available."
    )


# ============================================================
# ERROR SUMMARY
# ============================================================

st.divider()

st.subheader("Error Analysis Summary")

if isinstance(error_summary, dict):

    # Display structured JSON in a readable way.

    summary_rows = []

    for key, value in error_summary.items():

        if isinstance(
            value,
            (
                dict,
                list,
            ),
        ):
            continue

        summary_rows.append(
            {
                "Metric": (
                    str(key)
                    .replace("_", " ")
                    .title()
                ),
                "Value": value,
            }
        )

    if summary_rows:

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "No structured error summary is available."
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader("Interpretation")

st.markdown(
    """
- **Confusion pairs** show which waste categories the model
  most frequently mixes up.
- **Per-class performance** highlights categories where the
  classifier requires additional attention.
- Errors should be interpreted alongside the class distribution,
  image quality, and robustness results.
- A high-confidence prediction can still be incorrect, so
  confidence should not be treated as guaranteed correctness.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Smart Waste Classifier · Error Analysis · "
    "EfficientNetB0"
)