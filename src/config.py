# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Config."""


class TrainingConfig():
    """
    Config for training Info GAN
    """
    NAME = "infogan"
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE_DISC = 0.0002
    LEARNING_RATE_Q = 0.0002
    LEARNING_RATE_GEN = 0.0005
    ADAM_BETA = 0.5
    RELATIVE_LOSS = 1
    LATENT_NOISE_DIM = 62
    IMAGE_NOISE_SCALE = 0
    DATASET_NAME = "CRYPTO_PUNK" # Can also be "CRYPTO_PUNK"
    NUM_CATEGORIES = 10
    NUM_CONTINUOUS = 4
    DISCRIM_TRAIN_INTERVAL = 2 #How many generator training intervals are completed every discriminator update
    CONTIN_LOSS_SCALE = 0.3
    CAT_LOSS_SCALE = 1


class ReportingConfig():
    """
    Config for reporting / visulalization
    """
    SAVE_RESULTS_DIR = "trained_models_and_images"
    IMAGES_TO_SAVE = 10
    NUM_RAND_NOISE_VECT_PLOTS_TO_SAVE = 2
    PRINT_EVERY = 10
    CHECKPOINT_EVERY = 10
    SAVE_IMAGES_EVERY = 5