import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,BatchNormalization,Activation
from tensorflow.keras.initializers import RandomNormal
from matplotlib import pyplot


import numpy
from numpy import zeros,ones,expand_dims,hstack

from utils import load_dataset,generate_real_samples,generate_latent_points,generate_fake_samples
from models_config import define_discriminator_and_recognition,define_generator,define_gan


def summarize_performance(step, gen_model, gan_model, latent_dim, num_cat, num_samples=100):
    """

    Args:
        step: Step number
        gen_model: generator model
        gan_model: GAN model
        latent_dim: Latent variables dimension
        num_cat: number of categorical variables
        num_samples: number of samples that need to be created

    Returns:

    """
    # prepare fake samples
    X, _ = generate_fake_samples(gen_model, latent_dim, num_cat, num_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(num_samples):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = f'generated_plot_{(step+1)}.png'
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = f'generator_model_{(step+1)}.h5'
    gen_model.save(filename2)
    # save the gan model
    filename3 = f'gan_model_{(step+1)}.h5'
    gan_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))




# train the generator and discriminator
def train(gen_model, disc_model, gan_model, dataset, latent_dim, num_cat, num_epochs=100, batch_size=64):
	# calculate the number of batches per training epoch
	batch_per_epoch = int(dataset.shape[0] / batch_size)
	# calculate the number of training iterations
	num_steps = batch_per_epoch * num_epochs
	# calculate the size of half a batch of samples
	half_batch = int(batch_size / 2)
	# manually enumerate epochs
	for i in range(num_steps):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator and q model weights
		d_loss1 = disc_model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, num_cat, half_batch)
		# update discriminator model weights
		d_loss2 = disc_model.train_on_batch(X_fake, y_fake)
		# prepare points in latent space as input for the generator
		z_input, cat_codes = generate_latent_points(latent_dim, num_cat, batch_size)
		# create inverted labels for the fake samples
		y_gan = ones((batch_size, 1))
		# update the g via the d and q error
		_,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
		# summarize loss on this batch
		print(f'>{(i+1)}, d[{d_loss1, d_loss2}], g[{g_1}] q[{g_2}]')
		# evaluate the model performance every 'epoch'
		if (i+1) % (batch_per_epoch * 10) == 0:
			summarize_performance(i, gen_model, gan_model, latent_dim, num_cat)


# number of values for the categorical control code
n_cat = 10
# size of the latent space
latent_dim = 128
# create the discriminator
d_model, q_model = define_discriminator_and_recognition(n_cat)
# create the generator
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
# create the gan
gan_model = define_gan(g_model, d_model, q_model)
# load image data
dataset = load_dataset()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, n_cat)