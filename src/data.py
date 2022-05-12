"""
File to contain classes and functions relating to data ingress
"""
import numpy as np
import tensorflow as tf

def load_dataset(dataset_name):

    if dataset_name not in ["MNIST", "CRYPTO_PUNK"]:
        raise KeyError("Supported Dataset Names are [MNIST, CRYPTO_PUNK")

    if dataset_name == "CRYPTO_PUNK":
        with open("static/images/crypto.npy", "rb") as fp:
            x_train = np.load(fp)

    elif dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    dataset = x_train/(127.5) - 1
    return dataset