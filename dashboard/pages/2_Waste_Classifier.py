import streamlit as st
import requests

from components.styles import apply_styles


st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

apply_styles()


st.title("♻️ Waste Classifier")

st.caption(
    "Upload an image and classify it using the production API."
)

st.markdown("---")


API_URL = st.sidebar.text_input(
    "API URL",
    "http://localhost:8000",
)


uploaded_file = st.file_uploader(
    "Upload waste image",
    type=["jpg", "jpeg", "png", "webp"],
)


if uploaded_file:

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with col2:

        if st.button(
            "Classify Waste",
            type="primary",
            use_container_width=True,
        ):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=60,
                )

                if response.ok:

                    result = response.json()

                    st.success("Prediction completed.")

                    st.markdown(
                        '<div class="prediction-result">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "### Prediction"
                    )

                    st.markdown(
                        f'<div class="prediction-class">'
                        f'{result["predicted_class"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.metric(
                        "Confidence",
                        f'{result["confidence"] * 100:.2f}%',
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("---")

                    st.subheader(
                        "Top predictions"
                    )

                    for index, prediction in enumerate(
                        result["top_predictions"],
                        start=1,
                    ):

                        confidence = (
                            prediction["confidence"]
                            * 100
                        )

                        st.write(
                            f"**{index}. "
                            f"{prediction['class']}** — "
                            f"{confidence:.2f}%"
                        )

                        st.progress(
                            prediction["confidence"]
                        )

                else:

                    st.error(
                        response.json().get(
                            "detail",
                            "Prediction failed.",
                        )
                    )

            except requests.RequestException as exc:

                st.error(
                    f"Unable to connect to API: {exc}"
                )