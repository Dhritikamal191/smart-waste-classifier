# ============================================================
# SMART WASTE CLASSIFIER
# Grad-CAM Explainability
# ============================================================

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/waste_classifier_finetuned.keras"
)

DATASET_ROOT = Path(
    "data/raw/garbage_classification/garbage_classification"
)

OUTPUT_ROOT = Path(
    "reports/gradcam"
)

HEATMAP_ROOT = (
    OUTPUT_ROOT / "heatmaps"
)

METADATA_PATH = (
    OUTPUT_ROOT / "gradcam_metadata.json"
)

IMAGE_SIZE = (
    224,
    224
)

MAX_IMAGES_PER_CLASS = 2


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
# DIRECTORY SETUP
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

HEATMAP_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "SMART WASTE CLASSIFIER - GRAD-CAM EXPLAINABILITY"
)
print("=" * 70)

print(
    f"\nTensorFlow: {tf.__version__}"
)


# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices(
    "GPU"
)

if gpus:

    print(
        f"GPU detected: {gpus}"
    )

else:

    print(
        "GPU not detected. Running on CPU."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully."
)

print(
    f"\nModel input shape: {model.input_shape}"
)

print(
    f"Model output shape: {model.output_shape}"
)


print(
    "\nClasses:"
)

for index, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{index:2d} -> {class_name}"
    )


# ============================================================
# GET MODEL COMPONENTS
# ============================================================

backbone = model.get_layer(
    "efficientnetb0"
)

pooling = model.get_layer(
    "global_average_pooling"
)

classifier = model.get_layer(
    "classifier"
)


print(
    "\nEfficientNet backbone:"
)

print(
    f"Name: {backbone.name}"
)

print(
    f"Input: {backbone.input_shape}"
)

print(
    f"Output: {backbone.output_shape}"
)


# ============================================================
# GET TARGET LAYER
# ============================================================

target_layer = backbone.get_layer(
    "top_conv"
)

print(
    "\nGrad-CAM target layer:"
)

print(
    f"Name: {target_layer.name}"
)

print(
    f"Type: {target_layer.__class__.__name__}"
)

print(
    f"Output shape: {target_layer.output.shape}"
)


# ============================================================
# BUILD DIRECT BACKBONE GRADIENT MODEL
# ============================================================

# IMPORTANT:
#
# We intentionally construct the gradient model from the
# EfficientNet backbone itself.
#
# This avoids trying to connect an inner nested layer
# directly to the outer model input, which caused:
#
# "Output with path 0 is not connected to inputs"
#
# and later:
#
# "Gradients are None"
#
# The backbone is itself a Functional model, so this graph
# remains completely connected.

grad_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output,
    ],
    name="gradcam_backbone_model"
)


print(
    "\nGrad-CAM graph created successfully."
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # The saved model expects 0-255 RGB inputs.
    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    return (
        image_array,
        image_tensor
    )


# ============================================================
# GRAD-CAM
# ============================================================

def compute_gradcam(
    image_tensor
):

    with tf.GradientTape() as tape:

        conv_outputs, backbone_outputs = (
            grad_model(
                image_tensor,
                training=False
            )
        )

        # Reproduce the classification head manually.
        pooled = pooling(
            backbone_outputs
        )

        predictions = classifier(
            pooled,
            training=False
        )

        predicted_index = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            0,
            predicted_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "The Grad-CAM graph is not connected."
        )

    # Global-average-pool the gradients.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension.
    conv_output = conv_outputs[0]

    # Weighted combination of feature maps.
    heatmap = tf.reduce_sum(
        conv_output *
        pooled_gradients,
        axis=-1
    )

    # ReLU.
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize.
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap)
    )

    return (
        heatmap.numpy(),
        predictions.numpy()[0],
        int(predicted_index.numpy())
    )


# ============================================================
# HEATMAP COLOR IMAGE
# ============================================================

def save_heatmap(
    heatmap,
    image_array,
    output_path
):

    # Convert heatmap to uint8.
    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    )

    # Simple RGB heatmap.
    #
    # Red = stronger activation
    # Blue = weaker activation

    colored = np.zeros(
        (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
            3
        ),
        dtype=np.uint8
    )

    colored[:, :, 0] = np.uint8(
        heatmap_array
    )

    colored[:, :, 1] = np.uint8(
        heatmap_array * 0.35
    )

    colored[:, :, 2] = np.uint8(
        255 - heatmap_array
    )

    heatmap_rgb = Image.fromarray(
        colored
    )

    # Blend with original image.
    original = Image.fromarray(
        np.uint8(image_array)
    )

    overlay = Image.blend(
        original,
        heatmap_rgb,
        alpha=0.45
    )

    overlay.save(
        output_path
    )


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def select_images():

    selected = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = (
            DATASET_ROOT /
            class_name
        )

        if not class_dir.exists():

            print(
                f"WARNING: Missing class directory: "
                f"{class_dir}"
            )

            continue

        image_files = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }
            ]
        )

        image_files = image_files[
            :MAX_IMAGES_PER_CLASS
        ]

        for image_path in image_files:

            selected.append(
                (
                    class_index,
                    class_name,
                    image_path
                )
            )

    return selected


# ============================================================
# RUN GRAD-CAM
# ============================================================

selected_images = select_images()

print(
    f"\nImages selected for Grad-CAM: "
    f"{len(selected_images)}"
)


metadata = {
    "tensorflow_version": tf.__version__,
    "model": str(MODEL_PATH),
    "target_layer": target_layer.name,
    "target_layer_output_shape": [
        int(x)
        if x is not None
        else None
        for x in target_layer.output.shape
    ],
    "image_size": list(IMAGE_SIZE),
    "total_selected": len(
        selected_images
    ),
    "processed": 0,
    "correct": 0,
    "incorrect": 0,
    "images": [],
}


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, (
    true_index,
    true_class,
    image_path
) in enumerate(
    selected_images,
    start=1
):

    print(
        f"\n[{index}/{len(selected_images)}] "
        f"{image_path.name}"
    )

    try:

        image_array, image_tensor = (
            load_image(
                image_path
            )
        )

        heatmap, probabilities, predicted_index = (
            compute_gradcam(
                image_tensor
            )
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        is_correct = (
            predicted_index ==
            true_index
        )

        if is_correct:

            metadata["correct"] += 1

        else:

            metadata["incorrect"] += 1


        # ----------------------------------------------------
        # OUTPUT PATH
        # ----------------------------------------------------

        class_output_dir = (
            HEATMAP_ROOT /
            true_class
        )

        class_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        output_name = (
            image_path.stem +
            "_gradcam.jpg"
        )

        output_path = (
            class_output_dir /
            output_name
        )


        save_heatmap(
            heatmap,
            image_array,
            output_path
        )


        metadata["processed"] += 1

        metadata["images"].append(
            {
                "image": str(
                    image_path
                ),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "true_index": true_index,
                "predicted_index": predicted_index,
                "confidence": confidence,
                "correct": is_correct,
                "heatmap": str(
                    output_path
                ),
            }
        )


        print(
            f"True: {true_class}"
        )

        print(
            f"Predicted: {predicted_class}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

        print(
            f"Status: "
            f"{'CORRECT' if is_correct else 'INCORRECT'}"
        )

        print(
            f"Saved: {output_path}"
        )


    except Exception as error:

        print(
            f"ERROR: {error}"
        )


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GRAD-CAM COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nImages selected: "
    f"{len(selected_images)}"
)

print(
    f"Images processed: "
    f"{metadata['processed']}"
)

print(
    f"Correct predictions: "
    f"{metadata['correct']}"
)

print(
    f"Incorrect predictions: "
    f"{metadata['incorrect']}"
)

print(
    "\nReports saved to:"
)

print(
    f"  {OUTPUT_ROOT}"
)

print(
    "\nMetadata:"
)

print(
    f"  {METADATA_PATH}"
)

print(
    "\nHeatmaps:"
)

print(
    f"  {HEATMAP_ROOT}"
)

print(
    "\n" + "=" * 70
)