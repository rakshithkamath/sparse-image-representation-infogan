import numpy as np
from tensorflow.keras.utils import to_categorical
import pdb

def generate_real_samples(dataset, num_samples):
    """Randomly select real samples from dataset

    Labels will always be 1 for real samples

    Args:
        dataset (tf.DataSet or numpy.array): The real training samples
        num_samples (int): How many samples to pull

    Returns:
        tuple(X_train, Y_train)
    """
    index = np.random.randint(0, dataset.shape[0], num_samples)
    X = dataset[index]
    y = np.ones((num_samples, 1))
    return X, y

def generate_latent_points(latent_dim, cat_dim, num_samples):
    """Get random input to generator

    Input is made up of random normal noise and a 1 hot encoded vector
        with length cat_dim

    Args:
        latent_dim (_type_): _description_
        cat_dim (_type_): _description_
        num_samples (_type_): _description_

    Returns:
        z_input: np.array
        cat_codes: Needed as a label for the Q network
    """
    z_latent = np.random.normal(loc=0.0, scale=2, size=(num_samples, latent_dim))
    cat_codes = np.random.randint(0, cat_dim, num_samples)
    cat_codes = to_categorical(cat_codes, num_classes=cat_dim)
    z_input = np.hstack((z_latent, cat_codes))
    return [z_input, cat_codes]

def generate_fake_samples(generator, latent_dim, num_cat, num_samples):
    """Using the tf.generator network and noise, generate samples

    Args:
        generator (tf.keras.models.Model): The generator network
        latent_dim (int): Length of pure noise part of generator input
        num_cat (int): Number of categories for the latent variable
        num_samples (int): Number of samples

    Returns:
        tuple(X_train, y_train): Training data with y labels always as 0
    """
    z_input, _ = generate_latent_points(latent_dim, num_cat, num_samples)
    images = generator.predict(z_input)
    y = np.zeros((num_samples, 1))
    return images, y

def generate_all_cat_fake_samples(generator, latent_dim, cat_dim):
    """Using the tf.generator network and noise, generate samples for all variations of cat

    Args:
        generator (tf.keras.models.Model): The generator network
        latent_dim (int): Length of pure noise part of generator input
        num_cat (int): Number of categories for the latent variable
        num_samples (int): Number of samples

    Returns:
        tuple(X_train, y_train): Training data with y labels always as 0
    """
    num_samples = cat_dim
    # Use a larger scale for the noise
    z_latent = np.random.normal(loc=0.0, scale=2, size=(num_samples, latent_dim))
    cat_codes = np.arange(cat_dim)
    cat_codes = to_categorical(cat_codes, num_classes=cat_dim)
    z_input = np.hstack((z_latent, cat_codes))
    images = generator(z_input, training=True)
    return images