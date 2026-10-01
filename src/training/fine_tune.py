from pathlib import Path
import json

import tensorflow as tf

from src.data.tf_dataset import create_datasets
from src.models.waste_classifier import build_model


# ============================================================
# Configuration
# ============================================================

PHASE1_MODEL = Path(
    "models/waste_classifier_phase1.keras"
)

MODEL_DIRECTORY = Path(
    "models"
)

HISTORY_DIRECTORY = Path(
    "models/history"
)

BEST_MODEL_PATH = (
    MODEL_DIRECTORY
    / "waste_classifier_finetuned.keras"
)

HISTORY_PATH = (
    HISTORY_DIRECTORY
    / "finetune_history.json"
)

CLASS_WEIGHTS_FILE = Path(
    "data/class_weights.json"
)

EPOCHS = 10

FINE_TUNE_LAYERS = 40

LEARNING_RATE = 1e-5


# ============================================================
# Class weights
# ============================================================

def load_class_weights():

    with CLASS_WEIGHTS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return {
        int(class_id): float(weight)
        for class_id, weight
        in data["class_weights"].items()
    }


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

    return [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ============================================================
# Save history
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
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WASTE CLASSIFIER — PHASE 2 FINE-TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = create_datasets()

    print(
        f"\nClasses: {class_names}"
    )

    # --------------------------------------------------------
    # Load Phase 1 model
    # --------------------------------------------------------

    print(
        "\nLoading Phase 1 model..."
    )

    phase1_model = tf.keras.models.load_model(
        PHASE1_MODEL
    )

    print(
        f"Loaded: {PHASE1_MODEL}"
    )

    # --------------------------------------------------------
    # Build fine-tuning architecture
    # --------------------------------------------------------

    print(
        "\nBuilding fine-tuning model..."
    )

    model = build_model(
        num_classes=len(class_names),
        fine_tune=True,
        fine_tune_layers=FINE_TUNE_LAYERS,
    )

    # --------------------------------------------------------
    # Transfer Phase 1 weights
    # --------------------------------------------------------

    print(
        "\nTransferring Phase 1 weights..."
    )

    model.set_weights(
        phase1_model.get_weights()
    )

    # --------------------------------------------------------
    # Compile with very small LR
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    # --------------------------------------------------------
    # Show trainable parameters
    # --------------------------------------------------------

    trainable_count = sum(
        tf.size(variable).numpy()
        for variable in model.trainable_variables
    )

    total_count = sum(
        tf.size(variable).numpy()
        for variable in model.weights
    )

    print(
        f"\nTotal parameters: "
        f"{total_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Fine-tuned EfficientNet layers: "
        f"{FINE_TUNE_LAYERS}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = load_class_weights()

    print(
        "\nClass weights loaded."
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 2 FINE-TUNING")
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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

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

    print("\n")
    print("=" * 70)
    print("PHASE 2 FINE-TUNING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest epoch: "
        f"{best_epoch + 1}"
    )

    print(
        f"Best validation loss: "
        f"{history.history['val_loss'][best_epoch]:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{history.history['val_accuracy'][best_epoch]:.4f}"
    )

    print(
        f"\nBest model saved to:"
        f"\n{BEST_MODEL_PATH}"
    )

    print(
        f"\nHistory saved to:"
        f"\n{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()