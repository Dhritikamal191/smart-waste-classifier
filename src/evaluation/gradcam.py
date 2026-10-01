"""
PHASE 3 — GRAD-CAM EXPLAINABILITY

Generates Grad-CAM visualizations for selected high-confidence
misclassifications from the Phase 2 test evaluation.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/waste_classifier_finetuned.keras")

OUTPUT_DIR = Path("models/evaluation/phase3/gradcam")

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]


# ============================================================
# SELECTED ERROR CASES
# ============================================================

ERROR_CASES = [
    (
        "shoes",
        "clothes",
        "data/splits/test/shoes/shoes1826.jpg",
    ),
    (
        "green-glass",
        "brown-glass",
        "data/splits/test/green-glass/green-glass558.jpg",
    ),
    (
        "clothes",
        "shoes",
        "data/splits/test/clothes/clothes4611.jpg",
    ),
    (
        "white-glass",
        "plastic",
        "data/splits/test/white-glass/white-glass461.jpg",
    ),
    (
        "paper",
        "battery",
        "data/splits/test/paper/paper328.jpg",
    ),
    (
        "cardboard",
        "paper",
        "data/splits/test/cardboard/cardboard114.jpg",
    ),
    (
        "plastic",
        "clothes",
        "data/splits/test/plastic/plastic514.jpg",
    ),
    (
        "trash",
        "white-glass",
        "data/splits/test/trash/trash106.jpg",
    ),
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    """
    Load and preprocess image for the model.
    """

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")

    image_array = np.expand_dims(image_array, axis=0)

    return image_array, original


# ============================================================
# FIND EFFICIENTNET BACKBONE
# ============================================================

def find_efficientnet(model):
    """
    Locate the nested EfficientNet backbone.
    """

    print("\nSearching for EfficientNet backbone...")

    for layer in model.layers:

        print(
            f"Layer: {layer.name} | "
            f"Type: {layer.__class__.__name__}"
        )

        if (
            "efficientnet" in layer.name.lower()
            or "efficientnet" in layer.__class__.__name__.lower()
        ):
            print(f"\nSelected backbone: {layer.name}")

            return layer

    raise RuntimeError(
        "EfficientNet backbone could not be found."
    )


# ============================================================
# FIND TARGET CONVOLUTIONAL LAYER
# ============================================================

def find_target_layer(backbone):
    """
    Find the final useful convolutional feature layer
    inside EfficientNet.
    """

    candidates = []

    for layer in backbone.layers:

        try:

            output_shape = layer.output.shape

            if len(output_shape) == 4:

                candidates.append(layer)

        except Exception:
            continue

    if not candidates:

        raise RuntimeError(
            "No 4D convolutional feature layer found."
        )

    # Prefer the last 4D feature layer
    target_layer = candidates[-1]

    print(
        f"\nSelected Grad-CAM layer: "
        f"{target_layer.name}"
    )

    print(
        f"Layer output shape: "
        f"{target_layer.output.shape}"
    )

    return target_layer


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image,
    model,
    backbone,
    target_layer,
):
    """
    Generate Grad-CAM heatmap.

    This implementation works through the nested
    EfficientNet backbone.
    """

    image_tensor = tf.convert_to_tensor(
        image,
        dtype=tf.float32,
    )

    # --------------------------------------------------------
    # Create a model that exposes:
    #
    # EfficientNet input
    #        ↓
    # target convolutional layer
    #        ↓
    # backbone output
    #
    # --------------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output,
        ],
    )

    # --------------------------------------------------------
    # Build the classifier head after EfficientNet
    # --------------------------------------------------------

    classifier_layers = []

    found_backbone = False

    for layer in model.layers:

        if layer is backbone:
            found_backbone = True
            continue

        if found_backbone:
            classifier_layers.append(layer)

    if not classifier_layers:

        raise RuntimeError(
            "Could not identify classifier head "
            "after EfficientNet."
        )

    print(
        "\nClassifier head:"
    )

    for layer in classifier_layers:

        print(
            f"  {layer.name} "
            f"({layer.__class__.__name__})"
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = feature_model(
            image_tensor,
            training=False,
        )

        x = backbone_output

        for layer in classifier_layers:

            x = layer(x, training=False)

        predictions = x

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_class]

    # --------------------------------------------------------
    # Gradient of predicted class with respect
    # to convolutional features
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs,
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM graph could not be constructed."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1,
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap,
    )

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        float(predictions[0][predicted_class].numpy()),
    )


# ============================================================
# SAVE HEATMAP
# ============================================================

def save_heatmap(
    heatmap,
    original_image,
    output_path,
):
    """
    Save Grad-CAM heatmap overlaid on original image.
    """

    import matplotlib.pyplot as plt

    heatmap_resized = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        (
            original_image.shape[1],
            original_image.shape[0],
        )
    )

    heatmap_array = np.array(
        heatmap_resized
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(original_image)

    plt.imshow(
        heatmap_array,
        alpha=0.45,
        cmap="jet",
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 — GRAD-CAM EXPLAINABILITY")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Input shape: {model.input_shape}"
    )

    print(
        f"Output shape: {model.output_shape}"
    )

    # --------------------------------------------------------
    # Locate backbone
    # --------------------------------------------------------

    backbone = find_efficientnet(model)

    print(
        f"Backbone input shape: "
        f"{backbone.input_shape}"
    )

    print(
        f"Backbone output shape: "
        f"{backbone.output_shape}"
    )

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = find_target_layer(
        backbone
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    print(
        "\nGenerating Grad-CAM visualizations..."
    )

    successful = 0

    for index, (
        true_class,
        expected_prediction,
        image_path,
    ) in enumerate(
        ERROR_CASES,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(ERROR_CASES)}]"
        )

        print(
            f"True class:       {true_class}"
        )

        print(
            f"Expected mistake: {expected_prediction}"
        )

        print(
            f"Image:            {image_path}"
        )

        try:

            image, original = load_image(
                image_path
            )

            heatmap, predicted_class, confidence = (
                make_gradcam_heatmap(
                    image,
                    model,
                    backbone,
                    target_layer,
                )
            )

            predicted_name = CLASS_NAMES[
                predicted_class
            ]

            print(
                f"Model prediction: {predicted_name}"
            )

            print(
                f"Confidence:       {confidence:.4f}"
            )

            output_name = (
                f"{index:02d}_"
                f"{true_class}_"
                f"to_"
                f"{predicted_name}.png"
            )

            output_path = (
                OUTPUT_DIR /
                output_name
            )

            save_heatmap(
                heatmap,
                original,
                output_path,
            )

            print(
                f"Saved: {output_path}"
            )

            successful += 1

        except Exception as error:

            print(
                f"Grad-CAM failed for "
                f"{image_path}"
            )

            print(
                f"Error: {error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "PHASE 3 GRAD-CAM COMPLETE"
    )

    print("=" * 70)

    print(
        f"Visualizations generated: "
        f"{successful}/{len(ERROR_CASES)}"
    )

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()