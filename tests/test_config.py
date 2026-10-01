from src import config


def test_image_configuration():
    assert config.IMAGE_HEIGHT == 224
    assert config.IMAGE_WIDTH == 224
    assert config.IMAGE_SIZE == (224, 224)
    assert config.CHANNELS == 3


def test_training_configuration():
    assert config.BATCH_SIZE == 32
    assert config.EPOCHS == 20
    assert config.LEARNING_RATE == 1e-3
    assert config.RANDOM_SEED == 42


def test_data_split_configuration():
    total = (
        config.TRAIN_RATIO
        + config.VALIDATION_RATIO
        + config.TEST_RATIO
    )

    assert total == 1.0


def test_model_configuration():
    assert config.MODEL_NAME == "efficientnetb0"


def test_project_directories():
    assert config.PROJECT_ROOT.exists()
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.MODEL_DIR == config.PROJECT_ROOT / "models"
