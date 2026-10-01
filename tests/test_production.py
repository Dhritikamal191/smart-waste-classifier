import requests


API_URL = "http://localhost:8000"


def test_production_health():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_production_root():

    response = requests.get(
        f"{API_URL}/",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Smart Waste Classifier API"
    assert data["status"] == "running"


def test_production_model_info():

    response = requests.get(
        f"{API_URL}/model-info",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "classes" in data