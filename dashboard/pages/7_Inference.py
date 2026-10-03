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
# API CONFIGURATION
# ============================================================

DEFAULT_API_URL = "http://localhost:8000"

try:
    API_URL = st.secrets.get(
        STREAMLIT_API_URL
    )
except Exception:
    API_URL = DEFAULT_API_URL

API_URL = str(API_URL).rstrip("/")

st.sidebar.text_input(
    "Prediction API URL",
    value=API_URL,
    disabled=True,key="prediction_key"
)

st.sidebar.caption(
    f"Connected API: {API_URL}"
)

# ============================================================
# OPTIONAL SIDEBAR API OVERRIDE
# ============================================================

st.sidebar.title("♻️ Smart Waste")

st.sidebar.caption(
    "EfficientNetB0 · TensorFlow"
)

st.sidebar.markdown("---")


# The environment variable is used by default.
# This allows you to override the API URL manually when testing.
api_url_input = st.sidebar.text_input(
    "Prediction API URL",
    value=API_URL,key="api_key"
)

API_URL = api_url_input.strip().rstrip("/")


st.sidebar.markdown("---")

st.sidebar.caption(
    "Production ML Dashboard"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "♻️ Live Waste Classification"
)

st.caption(
    "Upload a waste image and send it to the trained "
    "EfficientNetB0 model through the FastAPI inference service."
)


st.divider()


# ============================================================
# API STATUS
# ============================================================

def check_api():
    """
    Check whether the FastAPI inference service is available.
    """

    if not API_URL:
        return False, None

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=10,
        )

        if response.status_code != 200:
            return False, None

        return True, response.json()

    except requests.RequestException:
        return False, None


api_online, health_data = check_api()


# ============================================================
# API STATUS DISPLAY
# ============================================================

if api_online:

    st.success(
        "● Prediction API is online"
    )

else:

    st.error(
        "● Prediction API is unavailable"
    )

    st.caption(
        f"Current API URL: `{API_URL}`"
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
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### Image information"
        )

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
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
# PREDICTION BUTTON
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


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if uploaded_file is None:

        st.warning(
            "Please upload an image first."
        )

        st.stop()


    # --------------------------------------------------------
    # FILE SIZE VALIDATION
    # --------------------------------------------------------

    MAX_FILE_SIZE = 10 * 1024 * 1024

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "The image is larger than the 10 MB API limit."
        )

        st.stop()


    # --------------------------------------------------------
    # SEND REQUEST TO FASTAPI
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
                timeout=120,
            )

        except requests.Timeout:

            st.error(
                "Prediction request timed out. "
                "The FastAPI service may be starting up."
            )

            st.stop()

        except requests.ConnectionError:

            st.error(
                "Unable to connect to the prediction API."
            )

            st.code(
                API_URL,
                language="text",
            )

            st.info(
                "Check that SMART_WASTE_API_URL points "
                "to your deployed FastAPI service."
            )

            st.stop()

        except requests.RequestException as exc:

            st.error(
                f"Prediction API request failed: {exc}"
            )

            st.stop()


    # --------------------------------------------------------
    # RESPONSE STATUS
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


    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        result = response.json()

    except ValueError:

        st.error(
            "The API returned an invalid JSON response."
        )

        st.stop()


    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    predicted_class = result.get(
        "predicted_class",
        "-",
    )


    try:

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 0.0


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

        display_class = (
            predicted_class
            .replace(
                "-",
                " ",
            )
            .title()
        )

        st.success(
            display_class
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


        for index, prediction in enumerate(
            top_predictions,
            start=1,
        ):

            class_name = prediction.get(
                "class",
                "-",
            )

            try:

                prediction_confidence = float(
                    prediction.get(
                        "confidence",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                prediction_confidence = 0.0


            display_name = (
                class_name
                .replace(
                    "-",
                    " ",
                )
                .title()
            )


            st.write(
                f"**{index}. {display_name}** — "
                f"{prediction_confidence * 100:.2f}%"
            )


            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        prediction_confidence,
                    ),
                )
            )


    # ========================================================
    # RAW API RESPONSE
    # ========================================================

    with st.expander(
        "View API response"
    ):

        st.json(
            result
        )


# ============================================================
# INFERENCE SERVICE INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Inference Service"
)


info_col1, info_col2, info_col3 = st.columns(
    3
)


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
# API ENDPOINT INFORMATION
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
