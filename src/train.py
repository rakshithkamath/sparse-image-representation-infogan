"""
Train a GAN
"""
import pdb
import argparse
import numpy as np
from config import GlobalConfig, TrainingConfig
from models import define_gan, define_discriminator_and_recognition, define_generator
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# from data import TrainingData
import tensorflow.keras.optimizers as optimizers


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
    z_latent = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, latent_dim))
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

def train(args):

    latent_dim = 62
    num_cat = 10
    batch_size = 64
    num_epochs = 100
    gen_input_size = latent_dim + num_cat


    # TODO: If path to a model is passed in, continue training using that model

    gen_model = define_generator(gen_input_size)
    disc_model, q_model = define_discriminator_and_recognition(num_cat)
    gan_model = define_gan(gen_model, disc_model, q_model)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    dataset = x_train
    
    # compile models
    disc_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5))
    opt = optimizers.Adam(lr=args.learning_rate, beta_1=args.adam_beta)
    gan_model.compile(loss=["binary_crossentropy", "categorical_crossentropy"], optimizer=opt)
    
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    num_steps = batch_per_epoch * num_epochs
    # manually enumerate epochs
    for i in range(num_steps):

        # Discriminator Training
        X_real, y_real = generate_real_samples(dataset, int(batch_size / 2))
        X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, num_cat, int(batch_size / 2))
        d_loss1 = disc_model.train_on_batch(X_real, y_real)
        d_loss2 = disc_model.train_on_batch(X_fake, y_fake)

        # Generator Training
        z_input, cat_codes = generate_latent_points(latent_dim, num_cat, batch_size)
        y_gan = np.ones((batch_size, 1))
        _, g_1, g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])

        print(f'i={(i+1)}, Disc Loss (real, fake)=({d_loss1:.3f} {d_loss2:.3f}), Gen Loss:{g_1:.3f} Q Loss {g_2:.3f}')

        # # evaluate the model performance every 'epoch'
        # if (i+1) % (batch_per_epoch * 10) == 0:
        #     summarize_performance(i, gen_model, gan_model, latent_dim, num_cat)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, default=TrainingConfig.NUM_EPOCHS,
                        help='Number of epochs to train.')
    parser.add_argument('--learning_rate', required=False, default=TrainingConfig.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--adam_beta', required=False, default=TrainingConfig.ADAM_BETA,
                        help='Beta')
    parser.add_argument('--model_dir', required=False,
                        default=GlobalConfig.get('MODEL_DIR'))
    args = parser.parse_args()

    train(args)