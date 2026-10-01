from pathlib import Path
import streamlit as st

from components.styles import apply_styles
from components.data import load_dashboard_data


st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

ROOT = Path(__file__).resolve().parent.parent

data = load_dashboard_data(ROOT)

st.sidebar.title("♻️ Smart Waste")
st.sidebar.caption("EfficientNetB0 · TensorFlow")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Dashboard**

    • Overview  
    • Waste Classifier  
    • Model Performance  
    • Robustness  
    • Error Analysis  
    • Explainability
    • Inference
    """
)

st.sidebar.markdown("---")

st.sidebar.caption("Smart Waste Classifier")
st.sidebar.caption("Production ML Dashboard")


st.title("♻️ Smart Waste Classifier")

st.markdown(
    """
    ### AI-powered waste classification

    Production-oriented dashboard for the fine-tuned **EfficientNetB0**
    waste classification system.
    """
)

st.markdown("---")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Waste Classes",
        data["num_classes"],
    )

with col2:
    st.metric(
        "Model",
        "EfficientNetB0",
    )

with col3:
    st.metric(
        "Framework",
        "TensorFlow",
    )

with col4:
    st.metric(
        "API",
        "FastAPI",
    )


st.markdown("---")

st.info(
    "Use the navigation menu on the left to explore model performance, "
    "robustness testing, error analysis, explainability, and live prediction."
)

st.markdown(
    """
    ### Project capabilities

    - 🧠 Fine-tuned EfficientNetB0 classifier
    - 📊 Multi-class waste classification
    - ⚡ FastAPI inference service
    - 🔍 Robustness evaluation
    - 🧪 Error and confusion analysis
    - 🔥 Grad-CAM explainability
    - 📈 Model performance monitoring
    - 🖥️ Interactive Streamlit dashboard
    """
)