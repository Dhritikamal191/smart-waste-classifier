import streamlit as st
import plotly.express as px

from components.data import load_dashboard_data
from components.styles import apply_styles


st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide",
)

apply_styles()

root = __import__("pathlib").Path(__file__).resolve().parents[2]
data = load_dashboard_data(root)


st.title("📊 Project Overview")

st.caption(
    "Smart Waste Classifier · Model and evaluation overview"
)

st.markdown("---")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Classes",
        data["num_classes"],
    )

with col2:
    st.metric(
        "Architecture",
        "EfficientNetB0",
    )

with col3:
    st.metric(
        "Input Size",
        "224 × 224",
    )

with col4:
    st.metric(
        "Inference API",
        "FastAPI",
    )


st.markdown("---")


st.subheader("Model evaluation")

report = data["clean_report"]

if report:

    accuracy = report.get("accuracy")

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        if accuracy is not None:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%",
            )

    with c2:
        st.metric(
            "Macro F1",
            f"{macro.get('f1-score', 0) * 100:.2f}%",
        )

    with c3:
        st.metric(
            "Weighted F1",
            f"{weighted.get('f1-score', 0) * 100:.2f}%",
        )

else:
    st.info(
        "Clean classification report not available."
    )


st.markdown("---")

st.subheader("Robustness overview")

df = data["robustness_results"]

if not df.empty:

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "Robustness results are not available."
    )