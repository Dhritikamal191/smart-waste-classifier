from pathlib import Path
import json

import tensorflow as tf

from src.data.tf_dataset import create_datasets
from src.models.waste_classifier import (
    build_model,
    compile_model,
)


# ============================================================
# Configuration
# ============================================================

CLASS_WEIGHTS_FILE = Path(
    "data/class_weights.json"
)

MODEL_DIRECTORY = Path(
    "models"
)

HISTORY_DIRECTORY = Path(
    "models/history"
)

BEST_MODEL_PATH = (
    MODEL_DIRECTORY
    / "waste_classifier_phase1.keras"
)

HISTORY_PATH = (
    HISTORY_DIRECTORY
    / "phase1_history.json"
)

EPOCHS = 15


# ============================================================
# Load class weights
# ============================================================

def load_class_weights():

    with CLASS_WEIGHTS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight
        in data["class_weights"].items()
    }

    return class_weights


# ============================================================
# Callbacks
# ============================================================

def create_callbacks():

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    return callbacks


# ============================================================
# Save training history
# ============================================================

def save_history(history):

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history_data,
            file,
            indent=2,
        )


# ============================================================
# Training
# ============================================================

def main():

    print("=" * 70)
    print("WASTE CLASSIFIER — PHASE 1 TRAINING")
    print("=" * 70)

    print("\nLoading TensorFlow datasets...")

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = create_datasets()

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    print(
        f"Classes: "
        f"{class_names}"
    )

    print("\nLoading class weights...")

    class_weights = load_class_weights()

    print(
        f"Class weights: "
        f"{class_weights}"
    )

    print("\nBuilding model...")

    model = build_model(
        num_classes=len(class_names)
    )

    model = compile_model(
        model
    )

    print("\nModel ready.")

    model.summary()

    callbacks = create_callbacks()

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history(
        history
    )

    print("\n")
    print("=" * 70)
    print("PHASE 1 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest model saved to:"
        f"\n{BEST_MODEL_PATH}"
    )

    print(
        f"\nTraining history saved to:"
        f"\n{HISTORY_PATH}"
    )

    print("\nBest validation metrics:")

    best_epoch = min(
        range(
            len(
                history.history[
                    "val_loss"
                ]
            )
        ),
        key=lambda index:
        history.history[
            "val_loss"
        ][index],
    )

    print(
        f"Epoch: {best_epoch + 1}"
    )

    print(
        f"Validation loss: "
        f"{history.history['val_loss'][best_epoch]:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{history.history['val_accuracy'][best_epoch]:.4f}"
    )


if __name__ == "__main__":
    main()