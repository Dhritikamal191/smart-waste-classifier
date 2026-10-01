from pathlib import Path
import os

import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Live Inference | Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = (
    os.getenv(
        "SMART_WASTE_API_URL",
        "http://localhost:8000",
    )
)

API_URL = st.sidebar.text_input(
    "Prediction API URL",
    value=DEFAULT_API_URL,
).rstrip("/")


# ============================================================
# PAGE HEADER
# ============================================================

st.title("♻️ Live Waste Classification")

st.caption(
    "Upload a waste image and send it to the trained "
    "EfficientNetB0 model through the FastAPI inference service."
)


# ============================================================
# API STATUS
# ============================================================

def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code != 200:

            return False, None

        return True, response.json()

    except requests.RequestException:

        return False, None


api_online, health_data = check_api()


if api_online:

    st.success(
        "● Prediction API is online"
    )

else:

    st.error(
        "● Prediction API is unavailable"
    )


# ============================================================
# MODEL STATUS
# ============================================================

if health_data:

    model_loaded = health_data.get(
        "model_loaded",
        False,
    )

    if model_loaded:

        st.info(
            "Model loaded and ready for inference."
        )

    else:

        st.warning(
            "API is running, but the trained model is not loaded."
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

st.subheader(
    "Upload Image"
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    help=(
        "Supported formats: JPG, JPEG, PNG and WEBP. "
        "Maximum file size: 10 MB."
    ),
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            width="stretch",
        )

    with col2:

        st.markdown(
            "### Image information"
        )

        file_size_mb = (
            uploaded_file.size
            / (
                1024
                * 1024
            )
        )

        st.metric(
            "File size",
            f"{file_size_mb:.2f} MB",
        )

        st.write(
            f"**Filename:** `{uploaded_file.name}`"
        )

        st.write(
            f"**Type:** `{uploaded_file.type}`"
        )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Classify Waste",
    type="primary",
    use_container_width=True,
    disabled=(
        uploaded_file is None
        or not api_online
    ),
)


if predict_button:

    if uploaded_file is None:

        st.warning(
            "Please upload an image first."
        )

        st.stop()


    # --------------------------------------------------------
    # SIZE VALIDATION
    # --------------------------------------------------------

    if uploaded_file.size > (
        10 * 1024 * 1024
    ):

        st.error(
            "The image is larger than the 10 MB API limit."
        )

        st.stop()


    # --------------------------------------------------------
    # SEND REQUEST
    # --------------------------------------------------------

    with st.spinner(
        "Running EfficientNetB0 inference..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
                timeout=60,
            )

        except requests.Timeout:

            st.error(
                "Prediction request timed out."
            )

            st.stop()

        except requests.RequestException as exc:

            st.error(
                f"Unable to connect to the prediction API: {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_data = response.json()

            detail = error_data.get(
                "detail",
                "Prediction failed.",
            )

        except ValueError:

            detail = (
                "Prediction API returned an unexpected response."
            )

        st.error(
            f"Prediction failed: {detail}"
        )

        st.stop()


    try:

        result = response.json()

    except ValueError:

        st.error(
            "The API returned an invalid JSON response."
        )

        st.stop()


    # --------------------------------------------------------
    # EXTRACT RESULT
    # --------------------------------------------------------

    predicted_class = result.get(
        "predicted_class",
        "-",
    )

    confidence = float(
        result.get(
            "confidence",
            0.0,
        )
    )

    top_predictions = result.get(
        "top_predictions",
        [],
    )


    # ========================================================
    # RESULT DISPLAY
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )


    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    result_col1, result_col2 = st.columns(
        [1.4, 1]
    )


    with result_col1:

        st.markdown(
            "### Predicted Waste Class"
        )

        st.success(
            predicted_class
            .replace(
                "-",
                " ",
            )
            .title()
        )


    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
        )


    # --------------------------------------------------------
    # CONFIDENCE BAR
    # --------------------------------------------------------

    st.progress(
        max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )
    )


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    if top_predictions:

        st.subheader(
            "Top Predictions"
        )

        rows = []

        for index, prediction in enumerate(
            top_predictions,
            start=1,
        ):

            class_name = prediction.get(
                "class",
                "-",
            )

            prediction_confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

            rows.append(
                {
                    "Rank": index,
                    "Waste Class": (
                        class_name
                        .replace(
                            "-",
                            " ",
                        )
                        .title()
                    ),
                    "Confidence": (
                        f"{prediction_confidence * 100:.2f}%"
                    ),
                }
            )

        st.table(
            rows
        )


    # ========================================================
    # RAW RESPONSE
    # ========================================================

    with st.expander(
        "View API response"
    ):

        st.json(
            result
        )


# ============================================================
# API INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Inference Service"
)

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Backend",
        "FastAPI",
    )

with info_col2:

    st.metric(
        "Model",
        "EfficientNetB0",
    )

with info_col3:

    st.metric(
        "Framework",
        "TensorFlow",
    )


# ============================================================
# ENDPOINT INFORMATION
# ============================================================

with st.expander(
    "API endpoints"
):

    st.code(
        f"""
GET  {API_URL}/
GET  {API_URL}/health
GET  {API_URL}/model-info
POST {API_URL}/predict
        """.strip(),
        language="text",
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Smart Waste Classifier · Live Inference · "
    "FastAPI · EfficientNetB0 · TensorFlow"
)