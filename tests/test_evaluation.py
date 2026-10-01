import numpy as np

from src.evaluation.evaluate import (
    calculate_confusion_matrix,
    calculate_classification_metrics,
)

import pytest

pytestmark = pytest.mark.local_ml

def test_confusion_matrix_shape():
    y_true = np.array([
        0,
        1,
        2,
        0,
    ])

    y_pred = np.array([
        0,
        1,
        1,
        2,
    ])

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert matrix.shape == (
        3,
        3,
    )


def test_confusion_matrix_values():
    y_true = np.array([
        0,
        1,
        2,
        0,
    ])

    y_pred = np.array([
        0,
        1,
        1,
        2,
    ])

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    expected = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
    ])

    np.testing.assert_array_equal(
        matrix,
        expected,
    )


def test_confusion_matrix_counts_all_samples():
    y_true = np.array([
        0,
        0,
        1,
        1,
        2,
        2,
    ])

    y_pred = np.array([
        0,
        1,
        1,
        1,
        2,
        0,
    ])

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert matrix.sum() == len(
        y_true
    )


def test_perfect_classification_metrics():
    matrix = np.array([
        [5, 0],
        [0, 5],
    ])

    metrics = calculate_classification_metrics(
        matrix
    )

    assert len(metrics) == 2

    for metric in metrics:
        assert metric["precision"] == 1.0
        assert metric["recall"] == 1.0
        assert metric["f1"] == 1.0


def test_classification_metrics_support():
    matrix = np.array([
        [3, 1],
        [2, 4],
    ])

    metrics = calculate_classification_metrics(
        matrix
    )

    assert metrics[0]["support"] == 4
    assert metrics[1]["support"] == 6


def test_classification_metrics_values():
    matrix = np.array([
        [3, 1],
        [2, 4],
    ])

    metrics = calculate_classification_metrics(
        matrix
    )

    # Class 0:
    # TP = 3
    # FP = 2
    # FN = 1

    expected_precision = 3 / 5
    expected_recall = 3 / 4
    expected_f1 = (
        2
        * expected_precision
        * expected_recall
        / (
            expected_precision
            + expected_recall
        )
    )

    assert np.isclose(
        metrics[0]["precision"],
        expected_precision,
    )

    assert np.isclose(
        metrics[0]["recall"],
        expected_recall,
    )

    assert np.isclose(
        metrics[0]["f1"],
        expected_f1,
    )


def test_metrics_are_finite():
    matrix = np.array([
        [5, 0, 0],
        [0, 0, 0],
        [0, 0, 4],
    ])

    metrics = calculate_classification_metrics(
        matrix
    )

    for metric in metrics:
        assert np.isfinite(
            metric["precision"]
        )

        assert np.isfinite(
            metric["recall"]
        )

        assert np.isfinite(
            metric["f1"]
        )

        assert metric["support"] >= 0
