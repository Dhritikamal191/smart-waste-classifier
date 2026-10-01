from src.api import app, METRICS


def test_metrics_endpoint(client):

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "successful_predictions" in data
    assert "failed_predictions" in data
    assert "success_rate" in data
    assert "failure_rate" in data
    assert "average_latency_ms" in data
    assert "average_confidence" in data
    assert "prediction_counts" in data


def test_metrics_reset_endpoint(client):

    response = client.post("/metrics/reset")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"