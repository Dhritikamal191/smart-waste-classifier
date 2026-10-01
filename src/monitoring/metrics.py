from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional


@dataclass
class PredictionMetrics:
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0

    total_latency_ms: float = 0.0
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None

    prediction_counts: Counter = field(
        default_factory=Counter
    )

    confidence_sum: float = 0.0

    def record_success(
        self,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
    ) -> None:

        self.total_predictions += 1
        self.successful_predictions += 1

        self.total_latency_ms += latency_ms

        if self.min_latency_ms is None:
            self.min_latency_ms = latency_ms
        else:
            self.min_latency_ms = min(
                self.min_latency_ms,
                latency_ms,
            )

        if self.max_latency_ms is None:
            self.max_latency_ms = latency_ms
        else:
            self.max_latency_ms = max(
                self.max_latency_ms,
                latency_ms,
            )

        self.prediction_counts[
            predicted_class
        ] += 1

        self.confidence_sum += confidence

    def record_failure(
        self,
        latency_ms: float = 0.0,
    ) -> None:

        self.total_predictions += 1
        self.failed_predictions += 1

        self.total_latency_ms += latency_ms

    @property
    def average_latency_ms(self) -> float:

        if self.total_predictions == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.total_predictions
        )

    @property
    def average_success_latency_ms(self) -> float:

        if self.successful_predictions == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.successful_predictions
        )

    @property
    def average_confidence(self) -> float:

        if self.successful_predictions == 0:
            return 0.0

        return (
            self.confidence_sum
            / self.successful_predictions
        )

    @property
    def success_rate(self) -> float:

        if self.total_predictions == 0:
            return 0.0

        return (
            self.successful_predictions
            / self.total_predictions
        )

    @property
    def failure_rate(self) -> float:

        if self.total_predictions == 0:
            return 0.0

        return (
            self.failed_predictions
            / self.total_predictions
        )

    def snapshot(self) -> Dict:

        return {
            "total_predictions":
                self.total_predictions,

            "successful_predictions":
                self.successful_predictions,

            "failed_predictions":
                self.failed_predictions,

            "success_rate":
                self.success_rate,

            "failure_rate":
                self.failure_rate,

            "average_latency_ms":
                self.average_latency_ms,

            "average_success_latency_ms":
                self.average_success_latency_ms,

            "min_latency_ms":
                self.min_latency_ms,

            "max_latency_ms":
                self.max_latency_ms,

            "average_confidence":
                self.average_confidence,

            "prediction_counts":
                dict(self.prediction_counts),
        }

    def reset(self) -> None:

        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0

        self.total_latency_ms = 0.0

        self.min_latency_ms = None
        self.max_latency_ms = None

        self.prediction_counts.clear()

        self.confidence_sum = 0.0


class PredictionTimer:

    def __init__(self):
        self.start_time = None

    def start(self) -> None:
        self.start_time = perf_counter()

    def stop(self) -> float:

        if self.start_time is None:
            return 0.0

        elapsed = (
            perf_counter()
            - self.start_time
        )

        self.start_time = None

        return elapsed * 1000