import tensorflow as tf


NUM_CLASSES = 12
IMAGE_SIZE = (224, 224, 3)


def build_model(
    num_classes=NUM_CLASSES,
    input_shape=IMAGE_SIZE,
    fine_tune=False,
    fine_tune_layers=40,
):
    """
    Build EfficientNetB0 transfer-learning model.

    Parameters
    ----------
    num_classes:
        Number of output classes.

    input_shape:
        Input image shape.

    fine_tune:
        If False, the entire EfficientNet backbone is frozen.

        If True, the final `fine_tune_layers` layers of
        EfficientNet are made trainable.

    fine_tune_layers:
        Number of EfficientNet layers to unfreeze.
    """

    inputs = tf.keras.Input(
        shape=input_shape,
        name="image",
    )

    # --------------------------------------------------------
    # Pretrained EfficientNetB0
    # --------------------------------------------------------

    base_model = (
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    )

    # --------------------------------------------------------
    # Transfer learning / fine-tuning
    # --------------------------------------------------------

    if not fine_tune:

        # Phase 1:
        # Entire pretrained backbone is frozen.

        base_model.trainable = False

    else:

        # Phase 2:
        # Freeze everything first.

        base_model.trainable = True

        for layer in base_model.layers:
            layer.trainable = False

        # Unfreeze only the final layers.

        for layer in base_model.layers[
            -fine_tune_layers:
        ]:
            layer.trainable = True

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    x = base_model(
        inputs,
        training=False,
    )

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="waste_classifier",
    )

    return model


def compile_model(
    model,
    learning_rate=1e-3,
):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


if __name__ == "__main__":

    print("=" * 70)
    print("WASTE CLASSIFIER MODEL")
    print("=" * 70)

    model = build_model()

    model = compile_model(
        model
    )

    model.summary()

    trainable_count = sum(
        tf.size(variable).numpy()
        for variable in model.trainable_variables
    )

    non_trainable_count = sum(
        tf.size(variable).numpy()
        for variable in model.non_trainable_variables
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_count:,}"
    )

    print(
        "\nModel construction successful."
    )