import os

import requests
import streamlit as st

from components.styles import apply_styles


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="wide",
)


# ============================================================
# STYLES
# ============================================================

apply_styles()


# ============================================================
# API CONFIGURATION
# ============================================================

# Local development:
#   http://localhost:8000
#
# Production:
#   Set STREAMLIT_API_URL in Streamlit Secrets
#
# Example:
#   STREAMLIT_API_URL = "https://your-fastapi-service.onrender.com"

DEFAULT_API_URL = "http://localhost:8000"


def get_api_url():
    """
    Get the FastAPI URL from Streamlit secrets/environment.

    Production:
        STREAMLIT_API_URL

    Local development:
        http://localhost:8000
    """

    # --------------------------------------------------------
    # Streamlit Secrets
    # --------------------------------------------------------

    try:

        secret_url = st.secrets.get(
            "STREAMLIT_API_URL",
            "",
        )

        if secret_url:

            return secret_url.rstrip("/")

    except Exception:

        pass


    # --------------------------------------------------------
    # Environment variable
    # --------------------------------------------------------

    environment_url = os.getenv(
        "STREAMLIT_API_URL",
        "",
    )

    if environment_url:

        return environment_url.rstrip("/")


    # --------------------------------------------------------
    # Local development fallback
    # --------------------------------------------------------

    return DEFAULT_API_URL


API_URL = get_api_url()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("♻️ Smart Waste")

st.sidebar.caption(
    "EfficientNetB0 · TensorFlow"
)

st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    **Waste Classifier**

    Upload a waste image and send it
    to the FastAPI inference service.
    """
)


st.sidebar.markdown("---")


# Show the API URL as read-only information instead of
# requiring the user to manually enter it every time.

st.sidebar.caption("Inference API")

st.sidebar.code(
    API_URL,
    language=None,
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("♻️ Waste Classifier")

st.caption(
    "Upload an image and classify it using the production API."
)

st.markdown("---")


# ============================================================
# API STATUS
# ============================================================

def check_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=10,
        )

        if response.ok:

            try:

                data = response.json()

            except ValueError:

                return False, (
                    "API returned an invalid JSON response."
                )

            if data.get("model_loaded") is True:

                return True, "API Online"

            return False, (
                "API is reachable, but the model is not loaded."
            )

        return False, (
            f"API returned HTTP {response.status_code}."
        )

    except requests.RequestException as exc:

        return False, str(exc)


# ============================================================
# HEALTH STATUS DISPLAY
# ============================================================

api_online, api_message = check_api_health()


if api_online:

    st.success(
        "🟢 FastAPI inference service is online."
    )

else:

    st.warning(
        "🟠 FastAPI inference service is not currently available."
    )

    with st.expander(
        "API connection details"
    ):

        st.write(
            f"**API URL:** `{API_URL}`"
        )

        st.write(
            api_message
        )

        st.info(
            "If you are running the project locally, start "
            "the FastAPI server on port 8000. If this dashboard "
            "is deployed, make sure STREAMLIT_API_URL points "
            "to your deployed FastAPI service."
        )


st.markdown("---")


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload waste image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    help=(
        "Upload a JPG, JPEG, PNG, or WEBP image."
    ),
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file:

    col1, col2 = st.columns(
        [1, 1],
        gap="large",
    )


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    with col1:

        st.subheader(
            "Uploaded image"
        )

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    with col2:

        st.subheader(
            "Classification"
        )


        classify_button = st.button(
            "♻️ Classify Waste",
            type="primary",
            use_container_width=True,
            disabled=not api_online,
        )


        if not api_online:

            st.caption(
                "Classification is disabled because the "
                "FastAPI inference service is unavailable."
            )


        if classify_button:

            with st.spinner(
                "Classifying image..."
            ):

                try:

                    # ----------------------------------------
                    # Prepare uploaded file
                    # ----------------------------------------

                    file_bytes = (
                        uploaded_file.getvalue()
                    )


                    files = {
                        "file": (
                            uploaded_file.name,
                            file_bytes,
                            uploaded_file.type,
                        )
                    }


                    # ----------------------------------------
                    # Send request to FastAPI
                    # ----------------------------------------

                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=60,
                    )


                    # ----------------------------------------
                    # Parse response
                    # ----------------------------------------

                    try:

                        result = response.json()

                    except ValueError:

                        st.error(
                            "The API returned an invalid response."
                        )

                        st.stop()


                    # ----------------------------------------
                    # Handle API error
                    # ----------------------------------------

                    if not response.ok:

                        detail = result.get(
                            "detail",
                            "Prediction failed.",
                        )

                        st.error(
                            f"Prediction failed: {detail}"
                        )

                        st.stop()


                    # ----------------------------------------
                    # Validate response
                    # ----------------------------------------

                    if "predicted_class" not in result:

                        st.error(
                            "The API response does not contain "
                            "a predicted class."
                        )

                        st.stop()


                    if "confidence" not in result:

                        st.error(
                            "The API response does not contain "
                            "a confidence value."
                        )

                        st.stop()


                    if "top_predictions" not in result:

                        st.error(
                            "The API response does not contain "
                            "top predictions."
                        )

                        st.stop()


                    # ----------------------------------------
                    # Prediction successful
                    # ----------------------------------------

                    st.success(
                        "Prediction completed successfully."
                    )


                    # ========================================
                    # MAIN RESULT
                    # ========================================

                    st.markdown(
                        '<div class="prediction-result">',
                        unsafe_allow_html=True,
                    )


                    st.markdown(
                        "### Prediction"
                    )


                    predicted_class = (
                        result["predicted_class"]
                    )


                    st.markdown(
                        f'<div class="prediction-class">'
                        f'{predicted_class}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


                    confidence = float(
                        result["confidence"]
                    )


                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%",
                    )


                    st.progress(
                        max(
                            0.0,
                            min(
                                1.0,
                                confidence,
                            ),
                        )
                    )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


                    # ========================================
                    # TOP PREDICTIONS
                    # ========================================

                    st.markdown("---")


                    st.subheader(
                        "Top predictions"
                    )


                    predictions = (
                        result["top_predictions"]
                    )


                    for index, prediction in enumerate(
                        predictions[:5],
                        start=1,
                    ):

                        prediction_class = (
                            prediction.get(
                                "class",
                                "Unknown",
                            )
                        )


                        prediction_confidence = float(
                            prediction.get(
                                "confidence",
                                0.0,
                            )
                        )


                        percentage = (
                            prediction_confidence
                            * 100
                        )


                        st.write(
                            f"**{index}. "
                            f"{prediction_class}** — "
                            f"{percentage:.2f}%"
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


                # ============================================
                # CONNECTION ERROR
                # ============================================

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Unable to connect to the FastAPI inference service."
                    )


                    st.info(
                        f"""
                        **Current API URL**

                        `{API_URL}`

                        Make sure the FastAPI service is running
                        and reachable from this Streamlit application.
                        """
                    )


                # ============================================
                # TIMEOUT
                # ============================================

                except requests.exceptions.Timeout:

                    st.error(
                        "The prediction request timed out."
                    )


                    st.info(
                        "The model may be taking longer than expected "
                        "to process the image. Please try again."
                    )


                # ============================================
                # OTHER REQUEST ERROR
                # ============================================

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Unable to connect to API: {exc}"
                    )


                # ============================================
                # UNEXPECTED ERROR
                # ============================================

                except Exception as exc:

                    st.error(
                        "An unexpected error occurred during prediction."
                    )


                    with st.expander(
                        "Technical details"
                    ):

                        st.exception(exc)


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Upload a waste image above to begin classification."
    )
