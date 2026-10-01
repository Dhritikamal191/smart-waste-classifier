from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Explainability | Smart Waste Classifier",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRADCAM_DIR = (
    PROJECT_ROOT
    / "reports"
    / "gradcam"
)

GRADCAM_METADATA = (
    GRADCAM_DIR
    / "gradcam_metadata.json"
)

GRADCAM_IMAGES = [
    path
    for path in GRADCAM_DIR.rglob("*")
    if path.suffix.lower()
    in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }
]


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔎 Model Explainability")

st.caption(
    "Grad-CAM visualization for understanding which image "
    "regions influence the EfficientNetB0 prediction."
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = None

if GRADCAM_METADATA.exists():

    try:

        with GRADCAM_METADATA.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as exc:

        st.warning(
            f"Unable to read Grad-CAM metadata: {exc}"
        )


# ============================================================
# OVERVIEW
# ============================================================

st.subheader(
    "Explainability Overview"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Explainability Method",
        "Grad-CAM",
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


# ============================================================
# EXPLANATION
# ============================================================

st.markdown(
    """
### What is Grad-CAM?

Grad-CAM highlights image regions that contribute most strongly
to a neural-network prediction.

For the Smart Waste Classifier, the visualization helps inspect
whether the EfficientNetB0 model is focusing on relevant visual
features of the waste item.

This provides a qualitative explanation of the prediction rather
than treating the model as a completely opaque system.
"""
)


# ============================================================
# METADATA
# ============================================================

if metadata is not None:

    st.divider()

    st.subheader(
        "Grad-CAM Metadata"
    )

    # --------------------------------------------------------
    # Handle dictionary metadata
    # --------------------------------------------------------

    if isinstance(
        metadata,
        dict,
    ):

        rows = []

        for key, value in metadata.items():

            if isinstance(
                value,
                (
                    dict,
                    list,
                ),
            ):

                value = json.dumps(
                    value,
                    indent=2,
                )

            rows.append(
                {
                    "Property": str(key),
                    "Value": str(value),
                }
            )

        if rows:

            metadata_df = pd.DataFrame(
                rows
            )

            st.dataframe(
                metadata_df,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # Handle list metadata
    # --------------------------------------------------------

    elif isinstance(
        metadata,
        list,
    ):

        st.json(
            metadata
        )


# ============================================================
# DISCOVER EXISTING VISUALIZATIONS
# ============================================================

st.divider()

st.subheader(
    "Grad-CAM Visualizations"
)

if not GRADCAM_IMAGES:

    st.info(
        "No Grad-CAM image files were found in "
        "`reports/gradcam/`."
    )

else:

    st.success(
        f"{len(GRADCAM_IMAGES)} explainability "
        "visualization(s) found."
    )

    # --------------------------------------------------------
    # SORT FILES
    # --------------------------------------------------------

    GRADCAM_IMAGES = sorted(
        GRADCAM_IMAGES,
        key=lambda path: path.name.lower(),
    )

    # --------------------------------------------------------
    # IMAGE GRID
    # --------------------------------------------------------

    for start in range(
        0,
        len(GRADCAM_IMAGES),
        3,
    ):

        row_images = GRADCAM_IMAGES[
            start:start + 3
        ]

        columns = st.columns(
            len(row_images)
        )

        for column, image_path in zip(
            columns,
            row_images,
        ):

            with column:

                st.image(
                    str(image_path),
                    caption=image_path.name,
                    width="stretch",
                )


# ============================================================
# INTERACTIVE IMAGE INSPECTION
# ============================================================

if GRADCAM_IMAGES:

    st.divider()

    st.subheader(
        "Inspect Visualization"
    )

    image_names = [
        image.name
        for image in GRADCAM_IMAGES
    ]

    selected_name = st.selectbox(
        "Select a Grad-CAM visualization",
        image_names,
    )

    selected_image = next(
        image
        for image in GRADCAM_IMAGES
        if image.name == selected_name
    )

    st.image(
        str(selected_image),
        caption=selected_image.name,
        width="stretch",
    )

    st.code(
        str(selected_image),
        language="text",
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "How to Interpret the Visualization"
)

st.markdown(
    """
### High-activation regions

Regions receiving stronger Grad-CAM activation represent areas
that contributed more strongly to the model's selected prediction.

For example, when classifying a plastic object, useful activation
may appear around the visible object rather than unrelated
background regions.

### What to look for

**Relevant attention**

The strongest activation overlaps with the actual waste object.

**Background dependence**

Strong activation appears primarily on irrelevant background
regions.

**Ambiguous objects**

Activation covers multiple visually similar objects or regions,
which can help explain difficult classifications.

**Consistent attention**

Similar examples of the same waste class produce attention on
similar visual structures.

Grad-CAM is therefore useful for qualitative model inspection,
error analysis, and communicating how a CNN arrives at a
prediction.
"""
)


# ============================================================
# LIMITATION
# ============================================================

with st.expander(
    "Important limitation"
):

    st.write(
        "Grad-CAM provides a visualization of feature importance "
        "within the network. It should not be interpreted as a "
        "formal causal explanation of the model's decision."
    )


# ============================================================
# SOURCE INFORMATION
# ============================================================

with st.expander(
    "Explainability implementation"
):

    gradcam_source = (
        PROJECT_ROOT
        / "src"
        / "evaluation"
        / "gradcam.py"
    )

    if gradcam_source.exists():

        st.code(
            str(gradcam_source),
            language="text",
        )

    else:

        st.write(
            "Grad-CAM source file path could not be located."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Smart Waste Classifier · Grad-CAM · "
    "EfficientNetB0 · TensorFlow"
)