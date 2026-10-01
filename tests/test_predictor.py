from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.inference.predictor import (
    CLASS_NAMES,
    NUM_CLASSES,
    MODEL_PATH,
    get_model_info,
    load_model,
    predict_image,
    preprocess_image,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def test_class_configuration():

    assert NUM_CLASSES == 12

    assert len(CLASS_NAMES) == 12

    assert "battery" in CLASS_NAMES
    assert "plastic" in CLASS_NAMES
    assert "white-glass" in CLASS_NAMES


def test_model_path_exists():

    assert MODEL_PATH.exists()

    assert MODEL_PATH.name == (
        "waste_classifier_finetuned.keras"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def test_model_info():

    info = get_model_info()

    assert info["model_name"] == "EfficientNetB0"

    assert info["model_file"] == (
        "waste_classifier_finetuned.keras"
    )

    assert info["image_size"] == [224, 224]

    assert info["num_classes"] == 12

    assert len(info["classes"]) == 12


# ============================================================
# MODEL LOADING
# ============================================================

def test_load_model():

    model = load_model()

    assert model is not None

    assert isinstance(
        model,
        tf.keras.Model,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def test_preprocess_image():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    assert image_path.exists()

    image = preprocess_image(
        image_path
    )

    assert image.shape == (
        1,
        224,
        224,
        3,
    )


def test_preprocess_missing_image():

    with pytest.raises(
        FileNotFoundError
    ):

        preprocess_image(
            "does_not_exist.jpg"
        )


# ============================================================
# PREDICTION VALIDATION
# ============================================================

def test_predict_invalid_top_k():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    with pytest.raises(
        ValueError
    ):

        predict_image(
            image_path,
            top_k=0,
        )


def test_predict_negative_top_k():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    with pytest.raises(
        ValueError
    ):

        predict_image(
            image_path,
            top_k=-1,
        )


def test_predict_non_integer_top_k():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    with pytest.raises(
        TypeError
    ):

        predict_image(
            image_path,
            top_k=3.5,
        )


# ============================================================
# PREDICTION
# ============================================================

def test_prediction_result():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=3,
    )

    assert isinstance(
        result,
        dict,
    )

    assert "image" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "top_predictions" in result


def test_prediction_class_is_valid():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=3,
    )

    assert result["predicted_class"] in CLASS_NAMES


def test_prediction_confidence():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=3,
    )

    confidence = result["confidence"]

    assert isinstance(
        confidence,
        float,
    )

    assert 0.0 <= confidence <= 1.0


def test_top_k_predictions():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=3,
    )

    predictions = result[
        "top_predictions"
    ]

    assert len(predictions) == 3

    for prediction in predictions:

        assert "class" in prediction
        assert "confidence" in prediction

        assert (
            prediction["class"]
            in CLASS_NAMES
        )

        assert (
            0.0
            <= prediction["confidence"]
            <= 1.0
        )


def test_top_k_cannot_exceed_class_count():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=100,
    )

    predictions = result[
        "top_predictions"
    ]

    assert len(predictions) == NUM_CLASSES


def test_top_predictions_are_sorted():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=5,
    )

    confidences = [
        prediction["confidence"]
        for prediction
        in result["top_predictions"]
    ]

    assert confidences == sorted(
        confidences,
        reverse=True,
    )


def test_top_prediction_matches_prediction():

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    result = predict_image(
        image_path,
        top_k=3,
    )

    first_prediction = (
        result["top_predictions"][0]
    )

    assert (
        first_prediction["class"]
        == result["predicted_class"]
    )

    assert np.isclose(
        first_prediction["confidence"],
        result["confidence"],
    )