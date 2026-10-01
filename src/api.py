from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference.predictor import (
    CLASS_NAMES,
    load_model,
    predict_image,
    get_model_info,
)

from src.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

from src.monitoring.metrics import (
    PredictionMetrics,
    PredictionTimer,
)

from src.monitoring.logger import logger


# ============================================================
# GLOBAL MODEL
# ============================================================

MODEL = None


# ============================================================
# GLOBAL MONITORING
# ============================================================

METRICS = PredictionMetrics()


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL

    logger.info(
        "Loading Smart Waste Classifier model..."
    )

    MODEL = load_model()

    logger.info(
        "Model loaded successfully."
    )

    yield

    MODEL = None

    logger.info(
        "Model unloaded."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Smart Waste Classifier API",
    description=(
        "Production-ready waste classification API "
        "using a fine-tuned EfficientNetB0 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Smart Waste Classifier API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "model_info": "/model-info",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():

    return get_model_info()


# ============================================================
# MONITORING METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return METRICS.snapshot()


# ============================================================
# RESET METRICS
# ============================================================

@app.post("/metrics/reset")
def reset_metrics():

    METRICS.reset()

    logger.info(
        "Prediction metrics reset."
    )

    return {
        "status": "success",
        "message": "Prediction metrics reset.",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
):

    request_id = str(uuid.uuid4())

    timer = PredictionTimer()

    timer.start()

    logger.info(
        f"[{request_id}] Prediction request received."
    )

    try:

        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if MODEL is None:

            raise HTTPException(
                status_code=503,
                detail="Model is not loaded.",
            )

        # ----------------------------------------------------
        # FILE TYPE VALIDATION
        # ----------------------------------------------------

        if not file.content_type:

            raise HTTPException(
                status_code=400,
                detail="File type could not be determined.",
            )

        if file.content_type not in ALLOWED_IMAGE_TYPES:

            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported image format. "
                    "Supported formats: JPEG, PNG, WEBP."
                ),
            )

        # ----------------------------------------------------
        # READ FILE
        # ----------------------------------------------------

        contents = await file.read()

        # ----------------------------------------------------
        # FILE SIZE VALIDATION
        # ----------------------------------------------------

        if len(contents) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Image file is too large. "
                    "Maximum size is 10 MB."
                ),
            )

        # ----------------------------------------------------
        # TEMPORARY UPLOAD DIRECTORY
        # ----------------------------------------------------

        upload_directory = Path(
            "tmp/uploads"
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # SAFE FILE NAME
        # ----------------------------------------------------

        original_filename = (
            file.filename or "upload"
        )

        safe_filename = Path(
            original_filename
        ).name

        file_path = (
            upload_directory
            / f"{request_id}_{safe_filename}"
        )

        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        with file_path.open("wb") as output_file:

            output_file.write(contents)

        # ----------------------------------------------------
        # VALIDATE IMAGE CONTENT
        # ----------------------------------------------------

        try:

            with Image.open(file_path) as image:

                image.verify()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded file is not a valid image."
                ),
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        result = predict_image(
            file_path,
            model=MODEL,
            top_k=3,
        )

        # ----------------------------------------------------
        # LATENCY
        # ----------------------------------------------------

        latency_ms = timer.stop()

        # ----------------------------------------------------
        # MONITORING
        # ----------------------------------------------------

        METRICS.record_success(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = PredictionResponse(
            filename=safe_filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
        )

        logger.info(
            f"[{request_id}] "
            f"Prediction: "
            f"{result['predicted_class']} "
            f"({result['confidence']:.4f}) "
            f"Latency: {latency_ms:.2f} ms"
        )

        return response

    except HTTPException:

        latency_ms = timer.stop()

        METRICS.record_failure(
            latency_ms=latency_ms
        )

        raise

    except Exception as exc:

        latency_ms = timer.stop()

        METRICS.record_failure(
            latency_ms=latency_ms
        )

        logger.exception(
            f"[{request_id}] "
            f"Prediction error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed due to an "
                "internal server error."
            ),
        )

    finally:

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        try:

            if "file_path" in locals():

                if file_path.exists():

                    file_path.unlink()

        except Exception as cleanup_error:

            logger.warning(
                f"[{request_id}] "
                f"Temporary file cleanup failed: "
                f"{cleanup_error}"
            )