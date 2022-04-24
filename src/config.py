# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Config."""

class GlobalConfig:
    """Config class for globals."""

    __conf = {
        "ROOT_DIR": "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/Sparse/project/sparse-image-representation-infogan/",
        "RANDOM_SEED": 42,
        # "TRAIN_IMAGE_DIR": "data/raw/",
        # "EXAMPLE_IMAGE_DIR": "data/example/",
        "MODEL_DIR": "trained_models.nosync",
        "LATEST_MODEL": "",
    }
    __setters = []

    @staticmethod
    def get(name):
        """Get config by name."""
        return GlobalConfig.__conf[name]

    @staticmethod
    def set(name, value):
        """Set config if config is settable."""
        if name in GlobalConfig.__setters:
            GlobalConfig.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")


class TrainingConfig():
    """
    Config for training Info GAN
    """
    NAME = "infogan"
    TRAIN_SAMPLE_SIZE = 2500*0.8
    VALID_SAMPLE_SIZE = 2500*0.2
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE_DISC = 0.0002
    LEARNING_RATE_GEN = 0.0002
    ADAM_BETA = 0.5
    RELATIVE_LOSS = 1
    LATENT_NOISE_DIM = 62


class InferenceConfig(TrainingConfig):
    OTHER_KEY = ""


class ReportingConfig():
    """
    Config for reporting / visulalization
    """
    IMAGES_TO_SAVE = 10
    # OUTPUT_IMAGE_WIDTH = 200
    # OUTPUT_IMAGE_HEIGHT = 400