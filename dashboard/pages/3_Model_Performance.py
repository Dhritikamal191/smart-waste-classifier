import streamlit as st
import plotly.express as px
from pathlib import Path

from components.data import load_dashboard_data
from components.styles import apply_styles


st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
)

apply_styles()

root = Path(__file__).resolve().parents[2]
data = load_dashboard_data(root)


st.title("📈 Model Performance")

st.caption(
    "Classification performance of the trained EfficientNetB0 model."
)

st.markdown("---")


report = data["clean_report"]

if not report:

    st.warning(
        "Classification report is not available."
    )

else:

    accuracy = report.get(
        "accuracy",
        0,
    )

    macro = report.get(
        "macro avg",
        {},
    )

    weighted = report.get(
        "weighted avg",
        {},
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%",
        )

    with c2:
        st.metric(
            "Macro Precision",
            f"{macro.get('precision', 0) * 100:.2f}%",
        )

    with c3:
        st.metric(
            "Macro F1",
            f"{macro.get('f1-score', 0) * 100:.2f}%",
        )

    st.markdown("---")

    rows = []

    for key, value in report.items():

        if not isinstance(value, dict):
            continue

        if "precision" not in value:
            continue

        rows.append(
            {
                "Class": key,
                "Precision": value["precision"],
                "Recall": value["recall"],
                "F1": value["f1-score"],
                "Support": value.get("support", 0),
            }
        )

    if rows:

        df = __import__("pandas").DataFrame(rows)

        st.subheader(
            "Per-class performance"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            df,
            x="Class",
            y=["Precision", "Recall", "F1"],
            barmode="group",
            template="plotly_dark",
            title="Precision, Recall and F1 by Class",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )