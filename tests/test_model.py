import tensorflow as tf

from src.models.waste_classifier import (
    build_model,
    compile_model,
    NUM_CLASSES,
    IMAGE_SIZE,
)


def test_model_creation():
    model = build_model()

    assert model is not None
    assert model.name == "waste_classifier"


def test_model_input_shape():
    model = build_model()

    assert model.input_shape == (
        None,
        224,
        224,
        3,
    )


def test_model_output_shape():
    model = build_model()

    assert model.output_shape == (
        None,
        NUM_CLASSES,
    )


def test_model_has_expected_class_count():
    model = build_model()

    assert model.output_shape[-1] == 12


def test_model_output_is_softmax():
    model = build_model()

    classifier = model.get_layer("classifier")

    assert classifier.activation == tf.keras.activations.softmax


def test_model_contains_expected_layers():
    model = build_model()

    layer_names = [
        layer.name
        for layer in model.layers
    ]

    assert "global_average_pooling" in layer_names
    assert "dropout" in layer_names
    assert "classifier" in layer_names


def test_backbone_frozen_without_fine_tuning():
    model = build_model(
        fine_tune=False
    )

    backbone = None

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.Model,
        ):
            backbone = layer
            break

    assert backbone is not None
    assert backbone.trainable is False


def test_fine_tuning_unfreezes_layers():
    model = build_model(
        fine_tune=True,
        fine_tune_layers=40,
    )

    backbone = None

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.Model,
        ):
            backbone = layer
            break

    assert backbone is not None
    assert backbone.trainable is True

    trainable_layers = [
        layer
        for layer in backbone.layers
        if layer.trainable
    ]

    assert len(trainable_layers) == 40


def test_model_compilation():
    model = build_model()

    compiled_model = compile_model(
        model,
        learning_rate=1e-3,
    )

    assert compiled_model.optimizer is not None
    assert compiled_model.loss is not None


def test_model_prediction_shape():
    model = build_model()

    images = tf.random.uniform(
        shape=(2, 224, 224, 3)
    )

    predictions = model.predict(
        images,
        verbose=0,
    )

    assert predictions.shape == (
        2,
        12,
    )


def test_model_predictions_are_probabilities():
    model = build_model()

    images = tf.random.uniform(
        shape=(2, 224, 224, 3)
    )

    predictions = model.predict(
        images,
        verbose=0,
    )

    assert tf.reduce_all(
        predictions >= 0
    )

    assert tf.reduce_all(
        predictions <= 1
    )

    sums = predictions.sum(
        axis=1
    )

    assert tf.reduce_all(
        tf.abs(sums - 1.0) < 1e-5
    )
