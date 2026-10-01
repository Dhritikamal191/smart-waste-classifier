import tensorflow as tf
import pytest

from src.data.tf_dataset import (
    load_dataset,
    prepare_train_dataset,
    prepare_eval_dataset,
)

pytestmark = pytest.mark.local_ml

def test_load_dataset(tmp_path):
    class_a = tmp_path / "class_a"
    class_b = tmp_path / "class_b"

    class_a.mkdir()
    class_b.mkdir()

    image = tf.random.uniform(
        shape=(224, 224, 3),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )

    image = tf.cast(
        image,
        tf.uint8,
    )

    encoded = tf.io.encode_jpeg(
        image
    )

    for index in range(2):
        tf.io.write_file(
            str(
                class_a
                / f"image_a_{index}.jpg"
            ),
            encoded,
        )

        tf.io.write_file(
            str(
                class_b
                / f"image_b_{index}.jpg"
            ),
            encoded,
        )

    dataset = load_dataset(
        tmp_path,
        shuffle=False,
    )

    assert dataset is not None

    assert sorted(
        dataset.class_names
    ) == [
        "class_a",
        "class_b",
    ]

    images, labels = next(
        iter(dataset)
    )

    assert images.shape[-3:] == (
        224,
        224,
        3,
    )

    assert labels.shape[0] == 4


def test_prepare_eval_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.zeros(
                (4, 224, 224, 3),
                dtype=tf.float32,
            ),
            tf.zeros(
                (4,),
                dtype=tf.int32,
            ),
        )
    ).batch(2)

    prepared = prepare_eval_dataset(
        dataset
    )

    assert prepared is not None

    images, labels = next(
        iter(prepared)
    )

    assert images.shape == (
        2,
        224,
        224,
        3,
    )

    assert labels.shape == (
        2,
    )


def test_prepare_train_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.zeros(
                (4, 224, 224, 3),
                dtype=tf.float32,
            ),
            tf.zeros(
                (4,),
                dtype=tf.int32,
            ),
        )
    ).batch(2)

    prepared = prepare_train_dataset(
        dataset
    )

    assert prepared is not None

    images, labels = next(
        iter(prepared)
    )

    assert images.shape == (
        2,
        224,
        224,
        3,
    )

    assert labels.shape == (
        2,
    )
