import pytest
from src.monitoring.metrics import (
    PredictionMetrics,
    PredictionTimer,
)


def test_initial_metrics():
    metrics = PredictionMetrics()

    assert metrics.total_predictions == 0
    assert metrics.successful_predictions == 0
    assert metrics.failed_predictions == 0
    assert metrics.average_latency_ms == 0.0
    assert metrics.average_confidence == 0.0


def test_record_success():

    metrics = PredictionMetrics()

    metrics.record_success(
        predicted_class="plastic",
        confidence=0.95,
        latency_ms=100.0,
    )

    assert metrics.total_predictions == 1
    assert metrics.successful_predictions == 1
    assert metrics.failed_predictions == 0

    assert metrics.success_rate == 1.0
    assert metrics.failure_rate == 0.0

    assert metrics.average_latency_ms == 100.0
    assert metrics.average_confidence == 0.95

    assert (
        metrics.prediction_counts["plastic"]
        == 1
    )


def test_record_multiple_predictions():

    metrics = PredictionMetrics()

    metrics.record_success(
        "plastic",
        0.90,
        100.0,
    )

    metrics.record_success(
        "metal",
        0.80,
        200.0,
    )

    assert metrics.total_predictions == 2
    assert metrics.successful_predictions == 2

    assert (
        metrics.average_latency_ms
        == 150.0
    )

    assert metrics.average_confidence == pytest.approx(
    0.85
    )

    assert (
        metrics.prediction_counts["plastic"]
        == 1
    )

    assert (
        metrics.prediction_counts["metal"]
        == 1
    )


def test_record_failure():

    metrics = PredictionMetrics()

    metrics.record_failure(
        latency_ms=50.0
    )

    assert metrics.total_predictions == 1
    assert metrics.successful_predictions == 0
    assert metrics.failed_predictions == 1

    assert metrics.success_rate == 0.0
    assert metrics.failure_rate == 1.0


def test_success_and_failure():

    metrics = PredictionMetrics()

    metrics.record_success(
        "paper",
        0.90,
        100.0,
    )

    metrics.record_failure(
        50.0
    )

    assert metrics.total_predictions == 2
    assert metrics.successful_predictions == 1
    assert metrics.failed_predictions == 1

    assert metrics.success_rate == 0.5
    assert metrics.failure_rate == 0.5

    assert (
        metrics.average_success_latency_ms
        == 150.0
    )


def test_snapshot():

    metrics = PredictionMetrics()

    metrics.record_success(
        "cardboard",
        0.92,
        80.0,
    )

    snapshot = metrics.snapshot()

    assert isinstance(snapshot, dict)

    assert (
        snapshot["total_predictions"]
        == 1
    )

    assert (
        snapshot["successful_predictions"]
        == 1
    )

    assert (
        snapshot["prediction_counts"]
        ["cardboard"]
        == 1
    )


def test_reset():

    metrics = PredictionMetrics()

    metrics.record_success(
        "plastic",
        0.95,
        100.0,
    )

    metrics.record_failure(50.0)

    metrics.reset()

    assert metrics.total_predictions == 0
    assert metrics.successful_predictions == 0
    assert metrics.failed_predictions == 0
    assert metrics.average_confidence == 0.0
    assert metrics.prediction_counts == {}


def test_timer():

    timer = PredictionTimer()

    timer.start()

    latency = timer.stop()

    assert latency >= 0.0


def test_timer_without_start():

    timer = PredictionTimer()

    assert timer.stop() == 0.0