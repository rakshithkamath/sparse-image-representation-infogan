"""Define the model architecture
"""
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import RandomNormal
from config import TrainingConfig
import pdb

def define_discriminator_and_recognition(cat_dim, num_continuous, input_shape=(28,28,1)):
    """
    Args:
    cat_dim:  number of categorical variables present in the dataset for our study
    input_shape: input dimension vectors for our dataset
    """
    in_image = layers.Input(shape=input_shape)
    d = layers.Conv2D(64, (4,4), strides=(2,2), padding="same",
        kernel_initializer=RandomNormal(stddev=0.02))(in_image)
    d = layers.LeakyReLU(alpha=0.1)(d)
    d = layers.Conv2D(128, (4,4), strides=(2,2), padding="same",
        kernel_initializer=RandomNormal(stddev=0.02))(d)
    d = layers.LeakyReLU(alpha=0.1)(d)
    d = layers.BatchNormalization()(d)
    d = layers.Flatten()(d)
    d = layers.Dense(1024)(d)
    d = layers.LeakyReLU(alpha=0.1)(d)
    out_classifier = layers.Dense(1, activation="sigmoid")(d)
    d_model = Model(in_image, out_classifier)
    q = layers.Dense(128)(d)
    q = layers.BatchNormalization()(q)
    q = layers.LeakyReLU(alpha=0.1)(q)
    cat_out_codes = layers.Dense(cat_dim, activation="softmax")(q)
    contin_out_codes = layers.Dense(num_continuous)(q)

    q_model = Model(in_image, outputs=[cat_out_codes, contin_out_codes])
    return d_model, q_model

def define_generator(input_shape):
    """Define the generator model

    Architecture taken from C.1 MNIST https://arxiv.org/pdf/1606.03657.pdf

    Args:
        input_shape (int): The length of the generator noise input

    Returns:
        tensorflow.keras.models.Model: Tensorflow model
    """
    input_dim = layers.Input(shape=(input_shape,))
    gen = layers.Dense(7*7*128, kernel_initializer=RandomNormal(stddev=0.02))(input_dim)
    gen = layers.Activation("relu")(gen)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Reshape((7, 7, 128))(gen)
    gen = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = layers.Activation("relu")(gen)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(gen)
    out_layer = layers.Activation("tanh")(gen)
    gen_model = Model(input_dim, out_layer)
    return gen_model

def define_gan(g_model, d_model, q_model):
    """Define the GAN network

    Architecture taken from C.1 MNIST https://arxiv.org/pdf/1606.03657.pdf


    Connect the output inputs and make weights in the discriminator 
    (some shared with the q model) as not trainable.

    Args:
        g_model (tensorflow.keras.models.Model): The generator model
        d_model (tensorflow.keras.models.Model): The discriminator model
        q_model (tensorflow.keras.models.Model): The Q model

    Returns:
        tensorflow.keras.models.Model: Tensorflow model of the joined system
    """
    for layer in d_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    d_output = d_model(g_model.output)
    q_output = q_model(g_model.output)


    if isinstance(q_output, list):
        joined_output = [d_output] + q_output
    else:
        joined_output = [d_output, q_output]

        

    model = Model(g_model.input, joined_output)
    return model

def define_discriminator_and_recognition_crypto_punk(cat_dim, num_continuous, input_shape=(24,24,3)):
    """
    Args:
    cat_dim:  number of categorical variables present in the dataset for our study
    input_shape: input dimension vectors for our dataset
    """
    in_image = layers.Input(shape=input_shape)
    d = layers.Conv2D(64, (4,4), strides=(2,2), padding="same",
        kernel_initializer=RandomNormal(stddev=0.02))(in_image)
    d = layers.LeakyReLU(alpha=0.1)(d)
    d = layers.Conv2D(128, (4,4), strides=(2,2), padding="same",
        kernel_initializer=RandomNormal(stddev=0.02))(d)
    d = layers.LeakyReLU(alpha=0.1)(d)
    d = layers.BatchNormalization()(d)
    d = layers.Flatten()(d)
    d = layers.Dense(1024)(d)
    d = layers.LeakyReLU(alpha=0.1)(d)
    out_classifier = layers.Dense(1, activation="sigmoid")(d)
    d_model = Model(in_image, out_classifier)
    q = layers.Dense(128)(d)
    q = layers.BatchNormalization()(q)
    q = layers.LeakyReLU(alpha=0.1)(q)
    cat_out_codes = layers.Dense(cat_dim, activation="softmax")(q)
    contin_out_codes = layers.Dense(num_continuous)(q)

    q_model = Model(in_image, outputs=[cat_out_codes, contin_out_codes])
    return d_model, q_model

def define_generator_crypto_punk(input_shape):
    """Define the generator model

    Architecture taken from C.1 MNIST https://arxiv.org/pdf/1606.03657.pdf

    Args:
        input_shape (int): The length of the generator noise input

    Returns:
        tensorflow.keras.models.Model: Tensorflow model
    """
    input_dim = layers.Input(shape=(input_shape,))
    gen = layers.Dense(3*6*6*128, kernel_initializer=RandomNormal(stddev=0.02))(input_dim)
    gen = layers.Activation("relu")(gen)
    gen = layers.Dense(6*6*128, kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = layers.Activation("relu")(gen)
    gen = layers.Dense(6*6*128, kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = layers.Activation("relu")(gen)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Reshape((6, 6, 128))(gen)
    gen = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = layers.Activation("relu")(gen)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding="same", kernel_initializer=RandomNormal(stddev=0.02))(gen)
    out_layer = layers.Activation("tanh")(gen)
    gen_model = Model(input_dim, out_layer)
    return gen_model

def define_gan_crypto_punk(g_model, d_model, q_model):
    """Define the GAN network

    Architecture taken from C.1 MNIST https://arxiv.org/pdf/1606.03657.pdf


    Connect the output inputs and make weights in the discriminator 
    (some shared with the q model) as not trainable.

    Args:
        g_model (tensorflow.keras.models.Model): The generator model
        d_model (tensorflow.keras.models.Model): The discriminator model
        q_model (tensorflow.keras.models.Model): The Q model

    Returns:
        tensorflow.keras.models.Model: Tensorflow model of the joined system
    """
    for layer in d_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    d_output = d_model(g_model.output)
    q_output = q_model(g_model.output)


    if isinstance(q_output, list):
        joined_output = [d_output] + q_output
    else:
        joined_output = [d_output, q_output]

    model = Model(g_model.input, joined_output)
    return model