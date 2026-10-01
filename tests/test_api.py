from pathlib import Path

from src.api import app

import pytest

# ============================================================
# HEALTH ENDPOINT
# ============================================================

@pytest.mark.local_ml
def test_health(client):
    """
    Verify that the API health endpoint works and
    the trained model is loaded during application startup.

    This test requires the locally trained model because
    the FastAPI lifespan loads the model during startup.
    """

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root(client):
    """
    Verify that the root endpoint is available.
    """

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    assert data["name"] == (
        "Smart Waste Classifier API"
    )

    assert data["status"] == "running"

    assert data["version"] == "1.0.0"


# ============================================================
# PREDICTION ENDPOINT — VALID IMAGE
# ============================================================

@pytest.mark.local_ml
def test_predict_valid_image(client):
    """
    Verify that the prediction endpoint accepts a valid
    waste image and returns the expected prediction structure.

    This test requires the local waste dataset and trained model.
    """

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/battery/battery1.jpg"
    )

    assert image_path.exists(), (
        f"Test image not found: {image_path}"
    )

    with image_path.open("rb") as image_file:

        response = client.post(
            "/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    # --------------------------------------------------------
    # Required response fields
    # --------------------------------------------------------

    assert "filename" in data
    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_predictions" in data

    # --------------------------------------------------------
    # Basic value validation
    # --------------------------------------------------------

    assert data["filename"] == image_path.name

    assert isinstance(
        data["predicted_class"],
        str,
    )

    assert 0.0 <= data["confidence"] <= 1.0

    assert isinstance(
        data["top_predictions"],
        list,
    )

    assert len(data["top_predictions"]) > 0


# ============================================================
# PREDICTION RESPONSE STRUCTURE
# ============================================================

@pytest.mark.local_ml
def test_predict_response_structure(client):
    """
    Verify the structure of the top prediction results.

    This test requires the local waste dataset and trained model.
    """

    image_path = Path(
        "data/raw/garbage_classification/"
        "garbage_classification/white-glass/white-glass1.jpg"
    )

    assert image_path.exists(), (
        f"Test image not found: {image_path}"
    )

    with image_path.open("rb") as image_file:

        response = client.post(
            "/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    predictions = data["top_predictions"]

    assert len(predictions) <= 3

    for prediction in predictions:

        assert "class" in prediction
        assert "confidence" in prediction

        assert isinstance(
            prediction["class"],
            str,
        )

        assert 0.0 <= prediction["confidence"] <= 1.0


# ============================================================
# INVALID FILE
# ============================================================

def test_predict_invalid_file(client):
    """
    Verify that the API rejects a non-image upload.

    This test does not require the trained model or dataset
    because validation happens before inference.
    """

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"This is not an image.",
                "text/plain",
            )
        },
    )

    assert response.status_code in (
        400,
        415,
        422,
    )


# ============================================================
# MISSING FILE
# ============================================================

def test_predict_missing_file(client):
    """
    Verify that the prediction endpoint rejects a request
    without an uploaded file.

    This test does not require the trained model or dataset.
    """

    response = client.post("/predict")

    assert response.status_code == 422


# ============================================================
# UNSUPPORTED IMAGE TYPE
# ============================================================

def test_predict_unsupported_file_type(client):
    """
    Verify that the API rejects an unsupported file type.
    """

    response = client.post(
        "/predict",
        files={
            "file": (
                "document.pdf",
                b"fake pdf content",
                "application/pdf",
            )
        },
    )

    assert response.status_code in (400, 415)

    data = response.json()

    assert "detail" in data


# ============================================================
# INVALID IMAGE CONTENT
# ============================================================

def test_predict_invalid_image_content(client):
    """
    Verify that an uploaded file claiming to be an image
    is rejected when its contents are not a valid image.

    This test does not require the trained model because
    image validation happens before inference.
    """

    response = client.post(
        "/predict",
        files={
            "file": (
                "fake.jpg",
                b"This is not actually an image.",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
