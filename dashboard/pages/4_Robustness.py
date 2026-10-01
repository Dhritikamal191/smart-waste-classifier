from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robustness | Smart Waste Classifier",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROBUSTNESS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "robustness"
)

PLOTS_DIR = (
    ROBUSTNESS_DIR
    / "plots"
)

RESULTS_PATH = (
    ROBUSTNESS_DIR
    / "robustness_results.csv"
)

PER_CLASS_PATH = (
    ROBUSTNESS_DIR
    / "per_class_robustness.csv"
)

CONFUSION_PAIRS_PATH = (
    ROBUSTNESS_DIR
    / "confusion_pairs.csv"
)

SUMMARY_PATH = (
    ROBUSTNESS_DIR
    / "robustness_summary.json"
)

ERROR_SUMMARY_PATH = (
    ROBUSTNESS_DIR
    / "robustness_error_summary.json"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Model Robustness")

st.caption(
    "Evaluation of model stability under image degradation "
    "including blur, contrast changes, and Gaussian noise."
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as exc:

        st.warning(
            f"Could not read {path.name}: {exc}"
        )

        return None


def load_json(path: Path):

    if not path.exists():
        return None

    try:

        import json

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


def find_column(
    dataframe,
    candidates,
):

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


# ============================================================
# LOAD REPORTS
# ============================================================

results = load_csv(
    RESULTS_PATH
)

per_class = load_csv(
    PER_CLASS_PATH
)

confusion_pairs = load_csv(
    CONFUSION_PAIRS_PATH
)

summary = load_json(
    SUMMARY_PATH
)

error_summary = load_json(
    ERROR_SUMMARY_PATH
)


# ============================================================
# CHECK DATA
# ============================================================

if (
    results is None
    and per_class is None
    and confusion_pairs is None
    and summary is None
):

    st.error(
        "No robustness evaluation results were found."
    )

    st.info(
        f"Expected reports inside:\n"
        f"{ROBUSTNESS_DIR}"
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

st.subheader("Robustness Overview")

if results is not None and not results.empty:

    df = results.copy()

    condition_column = find_column(
        df,
        [
            "condition",
            "corruption",
            "perturbation",
            "type",
        ],
    )

    severity_column = find_column(
        df,
        [
            "severity",
            "severity_level",
        ],
    )

    accuracy_column = find_column(
        df,
        [
            "accuracy",
            "accuracy_score",
        ],
    )

    f1_column = find_column(
        df,
        [
            "macro_f1",
            "f1",
            "macro_f1_score",
        ],
    )

    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    if accuracy_column:

        df[accuracy_column] = pd.to_numeric(
            df[accuracy_column],
            errors="coerce",
        )

    if f1_column:

        df[f1_column] = pd.to_numeric(
            df[f1_column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    accuracy_values = (
        df[accuracy_column].dropna()
        if accuracy_column
        else pd.Series(dtype=float)
    )

    f1_values = (
        df[f1_column].dropna()
        if f1_column
        else pd.Series(dtype=float)
    )

    best_accuracy = (
        accuracy_values.max()
        if not accuracy_values.empty
        else None
    )

    worst_accuracy = (
        accuracy_values.min()
        if not accuracy_values.empty
        else None
    )

    best_f1 = (
        f1_values.max()
        if not f1_values.empty
        else None
    )

    worst_f1 = (
        f1_values.min()
        if not f1_values.empty
        else None
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Best Accuracy",
            (
                f"{best_accuracy * 100:.2f}%"
                if best_accuracy is not None
                else "—"
            ),
        )

    with col2:

        st.metric(
            "Worst Accuracy",
            (
                f"{worst_accuracy * 100:.2f}%"
                if worst_accuracy is not None
                else "—"
            ),
        )

    with col3:

        st.metric(
            "Best Macro F1",
            (
                f"{best_f1 * 100:.2f}%"
                if best_f1 is not None
                else "—"
            ),
        )

    with col4:

        st.metric(
            "Worst Macro F1",
            (
                f"{worst_f1 * 100:.2f}%"
                if worst_f1 is not None
                else "—"
            ),
        )

else:

    st.info(
        "Summary metrics are unavailable."
    )


# ============================================================
# EXISTING ROBUSTNESS PLOTS
# ============================================================

st.divider()

st.subheader("Robustness Evaluation")

plot_files = [
    (
        "accuracy_by_severity.png",
        "Accuracy by Severity",
    ),
    (
        "accuracy_drop_by_condition.png",
        "Accuracy Drop by Condition",
    ),
    (
        "clean_vs_worst_condition.png",
        "Clean vs Worst Condition",
    ),
    (
        "macro_f1_by_severity.png",
        "Macro F1 by Severity",
    ),
]


for filename, title in plot_files:

    image_path = (
        PLOTS_DIR
        / filename
    )

    if image_path.exists():

        st.markdown(
            f"#### {title}"
        )

        st.image(
            str(image_path),
            width="stretch",
        )


# ============================================================
# RESULTS TABLE
# ============================================================

if results is not None and not results.empty:

    st.divider()

    st.subheader(
        "Robustness Results by Condition"
    )

    display_df = results.copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# INTERACTIVE ACCURACY ANALYSIS
# ============================================================

if (
    results is not None
    and not results.empty
    and condition_column
    and accuracy_column
):

    st.divider()

    st.subheader(
        "Interactive Robustness Analysis"
    )

    conditions = sorted(
        results[
            condition_column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_conditions = st.multiselect(
        "Select degradation conditions",
        conditions,
        default=conditions,
    )

    filtered = results[
        results[
            condition_column
        ]
        .astype(str)
        .isin(
            selected_conditions
        )
    ].copy()

    if not filtered.empty:

        import plotly.express as px

        if severity_column:

            filtered[severity_column] = pd.to_numeric(
                filtered[severity_column],
                errors="coerce",
            )

            fig = px.line(
                filtered,
                x=severity_column,
                y=accuracy_column,
                color=condition_column,
                markers=True,
                title=(
                    "Accuracy Under Increasing "
                    "Image Degradation"
                ),
            )

            fig.update_yaxes(
                tickformat=".0%",
            )

            fig.update_layout(
                height=500,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            fig = px.bar(
                filtered,
                x=condition_column,
                y=accuracy_column,
                title="Accuracy by Condition",
            )

            fig.update_yaxes(
                tickformat=".0%",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# PER-CLASS ROBUSTNESS
# ============================================================

st.divider()

st.subheader(
    "Per-Class Robustness"
)

if (
    per_class is not None
    and not per_class.empty
):

    df_class = per_class.copy()

    class_column = find_column(
        df_class,
        [
            "class",
            "class_name",
            "label",
        ],
    )

    metric_columns = [
        column
        for column in [
            "clean_accuracy",
            "robust_accuracy",
            "accuracy",
            "accuracy_drop",
            "macro_f1",
            "f1",
            "f1_score",
        ]
        if column in df_class.columns
    ]

    if class_column and metric_columns:

        selected_metric = st.selectbox(
            "Select class-level metric",
            metric_columns,
        )

        df_class[
            selected_metric
        ] = pd.to_numeric(
            df_class[
                selected_metric
            ],
            errors="coerce",
        )

        plot_df = df_class.dropna(
            subset=[
                selected_metric
            ]
        ).copy()

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

        import plotly.express as px

        fig = px.bar(
            plot_df,
            x=selected_metric,
            y="display_class",
            orientation="h",
            title=(
                f"Per-Class {selected_metric}"
            ),
        )

        fig.update_layout(
            height=600,
            yaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.dataframe(
        df_class,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Per-class robustness data is unavailable."
    )


# ============================================================
# CONFUSION PAIRS
# ============================================================

if (
    confusion_pairs is not None
    and not confusion_pairs.empty
):

    st.divider()

    st.subheader(
        "Robustness Confusion Pairs"
    )

    st.dataframe(
        confusion_pairs,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ROBUSTNESS HEATMAP
# ============================================================

heatmap_path = (
    PLOTS_DIR
    / "per_class_robustness_heatmap.png"
)

if heatmap_path.exists():

    st.divider()

    st.subheader(
        "Per-Class Robustness Heatmap"
    )

    st.image(
        str(heatmap_path),
        width="stretch",
    )


# ============================================================
# TOP CONFUSION PAIRS PLOT
# ============================================================

confusion_plot_path = (
    PLOTS_DIR
    / "top_confusion_pairs.png"
)

if confusion_plot_path.exists():

    st.divider()

    st.subheader(
        "Top Robustness Confusion Pairs"
    )

    st.image(
        str(confusion_plot_path),
        width="stretch",
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "What This Evaluation Shows"
)

st.markdown(
    """
### Robustness testing

The model is evaluated after controlled image degradation,
rather than only on clean test images.

The current evaluation includes:

- **Gaussian noise**
- **Blur**
- **Contrast degradation**
- Multiple degradation severity levels

### How to interpret the results

A model that performs well on clean images can still experience
performance degradation when image quality changes.

The robustness reports therefore help identify:

- conditions that cause the largest performance drop
- severity levels where degradation becomes significant
- waste classes that are particularly sensitive to image quality
- confusion patterns introduced by degraded images

These results complement the normal test-set evaluation rather
than replacing it.
"""
)


# ============================================================
# REPORT LOCATION
# ============================================================

with st.expander(
    "View report location"
):

    st.code(
        str(ROBUSTNESS_DIR),
        language="text",
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Smart Waste Classifier · Robustness Evaluation · "
    "EfficientNetB0"
)