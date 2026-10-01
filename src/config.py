from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

CHANNELS = 3


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3

RANDOM_SEED = 42


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "efficientnetb0"

MODEL_PATH = MODEL_DIR / "waste_classifier.keras"