"""Define the model architecture
"""
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import RandomNormal
from config import TrainingConfig

def define_discriminator_and_recognition(cat_dim, input_shape=(28,28,1)):
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
    out_codes = layers.Dense(cat_dim, activation="softmax")(q)
    q_model = Model(in_image, out_codes)
    return d_model, q_model

def define_generator(input_shape):
    """Define the generator model

    Architecture taken from C.1 MNIST https://arxiv.org/pdf/1606.03657.pdf

    Args:
        input_shape (int): The length of the generator noise input

    Returns:
        tensorflow.keras.models.Model: Tensorflow model
    """
    input_dim = tf.layers.Input(shape=(input_shape,))
    gen = tf.layers.Dense(7*7*128, kernel_initializer=RandomNormal(stddev=0.02))(input_dim)
    gen = tf.layers.Activation("relu")(gen)
    gen = tf.layers.BatchNormalization()(gen)
    gen = tf.layers.Reshape((7, 7, 128))(gen)
    gen = tf.layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(gen)
    gen = tf.layers.Activation("relu")(gen)
    gen = tf.layers.BatchNormalization()(gen)
    gen = tf.layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(gen)
    out_layer = tf.layers.Activation("tanh")(gen)
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
        if not isinstance(layer, tf.layers.BatchNormalization):
            layer.trainable = False
    d_output = d_model(g_model.output)
    q_output = q_model(g_model.output)
    model = Model(g_model.input, [d_output, q_output])
    return model