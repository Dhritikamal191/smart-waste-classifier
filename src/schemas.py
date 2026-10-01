from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TopPrediction(BaseModel):
    """Single class prediction."""

    model_config = ConfigDict(
        populate_by_name=True
    )

    class_name: str = Field(
        ...,
        alias="class",
        description="Predicted waste category.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence.",
    )


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    filename: str = Field(
        ...,
        description="Name of the uploaded image.",
    )

    predicted_class: str = Field(
        ...,
        description="Most likely waste category.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the predicted class.",
    )

    top_predictions: List[TopPrediction] = Field(
        ...,
        description="Top-k predictions.",
    )


class HealthResponse(BaseModel):
    """API health response."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Information about the deployed model."""

    model_name: str
    model_file: str
    image_size: List[int]
    num_classes: int
    classes: List[str]


class MetricsResponse(BaseModel):
    """Prediction monitoring metrics."""

    total_predictions: int
    successful_predictions: int
    failed_predictions: int

    success_rate: float

    average_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    average_confidence: float

    prediction_counts: dict[str, int]