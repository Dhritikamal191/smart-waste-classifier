from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import IMAGE_SIZE


# ============================================================
# MODEL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "waste_classifier_finetuned.keras"
)


MODEL_NAME = "EfficientNetB0"


# ============================================================
# CLASS NAMES
# ============================================================

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


NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the trained waste classification model.

    Returns
    -------
    tensorflow.keras.Model
        Loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the trained model does not exist.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():
    """
    Return metadata about the deployed model.
    """

    return {
        "model_name": MODEL_NAME,
        "model_file": MODEL_PATH.name,
        "image_size": list(IMAGE_SIZE),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and prepare an image for inference.

    Images are resized to the same dimensions used
    during model training.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input image.

    Returns
    -------
    tensorflow.Tensor
        Image tensor with shape (1, height, width, channels).
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image = tf.keras.utils.img_to_array(
        image
    )

    image = tf.expand_dims(
        image,
        axis=0,
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    image_path,
    model=None,
    top_k=3,
):
    """
    Predict the waste category of an image.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input image.

    model : tensorflow.keras.Model, optional
        Already-loaded model.

    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict
        Prediction results.
    """

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    top_k = min(
        top_k,
        NUM_CLASSES,
    )

    # --------------------------------------------------------
    # Load model if necessary
    # --------------------------------------------------------

    if model is None:
        model = load_model()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    probabilities = model.predict(
        image,
        verbose=0,
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if len(probabilities) != NUM_CLASSES:
        raise ValueError(
            "Model output does not match "
            f"the expected {NUM_CLASSES} classes."
        )

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    # --------------------------------------------------------
    # Top-k predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    top_predictions = []

    for index in top_indices:

        index = int(index)

        top_predictions.append(
            {
                "class": CLASS_NAMES[index],
                "confidence": float(
                    probabilities[index]
                ),
            }
        )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "image": str(image_path),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_prediction(result):
    """
    Print prediction results in a readable format.
    """

    print()
    print("=" * 60)
    print("WASTE CLASSIFICATION RESULT")
    print("=" * 60)

    print(
        f"Image      : {result['image']}"
    )

    print(
        f"Prediction : {result['predicted_class']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print()
    print("Top Predictions")
    print("-" * 60)

    for rank, prediction in enumerate(
        result["top_predictions"],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{prediction['class']:<15} "
            f"{prediction['confidence'] * 100:.2f}%"
        )

    print("=" * 60)


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Predict waste category "
            "from an image."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the image to classify.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of top predictions "
            "to display."
        ),
    )

    args = parser.parse_args()

    result = predict_image(
        args.image,
        top_k=args.top_k,
    )

    print_prediction(
        result
    )