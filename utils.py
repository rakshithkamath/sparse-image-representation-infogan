

import numpy
from numpy import zeros,ones,expand_dims,hstack
from numpy.random import randn,randint

import tensorflow as tf
from tensorflow.keras.utils import to_categorical




def load_dataset():
    #Need to write function so as to load the dataset




def generate_real_samples(dataset,num_samples):
    """
    Samples real dataset with class labels, that needs to be fed into the network
    """
    index = randint(0, dataset.shape[0], num_samples)
    # select images and labels
    X = dataset[index]
    # generate class labels
    y = ones((num_samples, 1))
    return X, y





def generate_latent_points(latent_dim, num_cat, num_samples):
    """
    generate points in latent space as input for the generator
    
    """
    # generate points in the latent space
    z_latent = randn(latent_dim * num_samples)
    # reshape into a batch of inputs for the network
    z_latent = z_latent.reshape(num_samples, latent_dim)

    # generate categorical codes
    cat_codes = randint(0, num_cat, num_samples)
    # one hot encode
    cat_codes = to_categorical(cat_codes, num_classes=num_cat)
    # concatenate latent points and control codes
    z_input = hstack((z_latent, cat_codes))
    return [z_input, cat_codes]
 





def generate_fake_samples(generator, latent_dim, num_cat, num_samples):
    """
    use the generator to generate n fake examples, with class labels
    """
    # generate points in latent space and control codes
    z_input, _ = generate_latent_points(latent_dim, num_cat, num_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((num_samples, 1))
    return images, y






