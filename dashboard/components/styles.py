import streamlit as st


def apply_styles():

    st.markdown(
        """
        <style>

        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(0, 208, 132, 0.08),
                    transparent 35%
                ),
                #0B0F14;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }

        h2, h3 {
            font-weight: 700 !important;
        }

        [data-testid="stMetric"] {
            background: #111820;
            border: 1px solid #202A35;
            border-radius: 16px;
            padding: 18px;
        }

        [data-testid="stMetricValue"] {
            font-weight: 800;
        }

        .dashboard-card {
            background: #111820;
            border: 1px solid #202A35;
            border-radius: 18px;
            padding: 22px;
            margin-bottom: 18px;
        }

        .dashboard-card h3 {
            margin-top: 0;
        }

        .section-label {
            color: #00D084;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .prediction-result {
            background: #111820;
            border: 1px solid #00D084;
            border-radius: 18px;
            padding: 25px;
        }

        .prediction-class {
            font-size: 2.1rem;
            font-weight: 800;
            color: #00D084;
        }

        footer {
            visibility: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )