"""
Train a GAN
"""
import pdb
import os
import argparse
from re import T
import time
import glob
# from black import TRANSFORMED_MAGICS
import numpy as np
from config import ReportingConfig
from visualization import summarize_performance_categorical, summarize_performance_categorical
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
from pathlib import Path


from config import GlobalConfig, TrainingConfig
from models import LOAD_MODEL_FUNC
from utils import generate_real_samples, generate_latent_points, generate_fake_samples
from visualization import summarize_performance_continuous
from data import load_dataset
# from data import TrainingData


def train(args):

    latent_dim = args.latent_dim
    num_categories = args.num_categories
    num_continuous = args.num_continuous
    disc_train_interval = args.disc_train_interval
    gen_input_size = latent_dim + num_categories + num_continuous

    if args.dataset_name not in ["MNIST", "CRYPTO_PUNK"]:
        raise KeyError("Supported Dataset Names are [MNIST, CRYPTO_PUNK")

    load_model_funcs = LOAD_MODEL_FUNC[args.dataset_name]
    gen_model = load_model_funcs["generator"](gen_input_size)
    disc_model, q_model = load_model_funcs["discriminator_recognition"](
        num_categories, num_continuous)
    # Compile discriminator before passing to define_gan as that will set some weights as non-trainible
    disc_model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(
        lr=args.learning_rate_disc))
    gan_model = load_model_funcs["gan"](gen_model, disc_model, q_model)

    dataset = load_dataset(args.dataset_name)

    # Make new sub folder for this particular run
    this_time_folder = os.path.join(
        args.model_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    Path(this_time_folder).mkdir(parents=True, exist_ok=True)

    if args.model_in_folder:
        this_time_folder = args.model_in_folder
        joined_path = os.path.join(
            args.model_in_folder, "*_generator_model.h5")
        poss_files = glob.glob(joined_path)
        latest_i = max([int(os.path.split(x)[1].split("_")[0])
                       for x in poss_files])
        gen_model.load_weights(os.path.join(
            args.model_in_folder, f"{latest_i}_generator_model.h5"))
        gan_model.load_weights(os.path.join(
            args.model_in_folder, f"{latest_i}_gan_model.h5"))
        disc_model.load_weights(os.path.join(
            args.model_in_folder, f"{latest_i}_disc_model.h5"))
        offset = latest_i
    else:
        offset = 0

    gen_loss_func1 = tf.keras.losses.BinaryCrossentropy()
    gen_loss_func_cat = tf.keras.losses.CategoricalCrossentropy()
    gen_optimizer = optimizers.Adam(lr=args.learning_rate_gen)
    q_optimizer = optimizers.Adam(lr=args.learning_rate_q)

    batches_per_epoch = int(dataset.shape[0] / args.batch_size)
    num_steps = batches_per_epoch * args.epochs

    d_loss1, d_loss2 = 0, 0
    for iter in range(num_steps):
        i = iter + offset

        if (iter+1) % disc_train_interval == 0:
            # Discriminator Training
            X_real, y_real = generate_real_samples(
                dataset, int(args.batch_size / 2))
            X_fake, y_fake = generate_fake_samples(
                gen_model, latent_dim, num_categories, num_continuous, int(args.batch_size / 2))
            if args.noise_scale:
                X_real += np.random.normal(0, args.noise_scale, size=X_real.shape)
                X_fake += np.random.normal(0,args.noise_scale, size=X_fake.shape)
            d_loss1 = disc_model.train_on_batch(X_real, y_real)
            d_loss2 = disc_model.train_on_batch(X_fake, y_fake)

        # Generator Training
        with tf.GradientTape() as g_tape, tf.GradientTape() as q_tape:
            z_input, cat_codes, contin_codes = generate_latent_points(
                latent_dim, num_categories, num_continuous, int(args.batch_size / 2))
            y_gan = np.ones((int(args.batch_size / 2), 1))
            g_tape.watch(gen_model.trainable_variables)
            q_tape.watch(q_model.trainable_variables)
            gan_disc_output, q_pred_cat_codes, contin_mu, contin_sigma = gan_model(
                z_input, training=True)
            # Use Gaussian distributions to represent the output
            # Note, I'm not sure if this works with multiple continuous variables.
            dist = tfp.distributions.Normal(loc=contin_mu, scale=contin_sigma)
            # Losses (negative log probability density function as we want to maximize the probability density function)
            q_loss_continuous = tf.reduce_mean(-dist.log_prob(contin_codes))
            gan_fooling_loss = gen_loss_func1(y_gan, gan_disc_output)
            q_loss_cat = gen_loss_func_cat(cat_codes, q_pred_cat_codes)
            q_loss = q_loss_cat*TrainingConfig.CAT_LOSS_SCALE + \
                     q_loss_continuous*TrainingConfig.CONTIN_LOSS_SCALE
            gen_loss = gan_fooling_loss + \
                     q_loss_cat*TrainingConfig.CAT_LOSS_SCALE + \
                     q_loss_continuous*TrainingConfig.CONTIN_LOSS_SCALE


        # Propogate loss for Generator and Q Network
        g_gradients = g_tape.gradient(gen_loss, gen_model.trainable_variables)
        q_gradients = q_tape.gradient(q_loss, q_model.trainable_variables)
        gen_optimizer.apply_gradients(
            zip(g_gradients, gen_model.trainable_variables))
        q_optimizer.apply_gradients(
            zip(q_gradients, q_model.trainable_variables))

        if (i+1) % args.print_every == 0:
            print(f"i={i+1}, Disc Loss (real, fake)=({d_loss1:.3f} {d_loss2:.3f}),"
                f" Gen Loss:{gan_fooling_loss:.3f} Q Loss cat {q_loss_cat:.3f} "
                f"Q Loss contin {q_loss_continuous:.3f}")

        if (i+1) % (batches_per_epoch * ReportingConfig.SAVE_IMAGES_EVERY) == 0:
            for j in range(ReportingConfig.NUM_RAND_NOISE_VECT_PLOTS_TO_SAVE):
                summarize_performance_continuous(
                    this_time_folder, i+j,
                    gen_model,
                    gan_model,
                    latent_dim,
                    num_categories,
                    num_continuous)
                summarize_performance_categorical(
                    this_time_folder, i+j,
                    gen_model,
                    gan_model,
                    latent_dim,
                    num_categories,
                    num_continuous)

        if (i+1) % ( ReportingConfig.CHECKPOINT_EVERY * batches_per_epoch) == 0:
            gen_model.save(os.path.join(this_time_folder,f"{(i+1)}_generator_model.h5"))
            gan_model.save(os.path.join(this_time_folder, f"{(i+1)}_gan_model.h5"))
            disc_model.save(os.path.join(this_time_folder, f"{(i+1)}_disc_model.h5"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=TrainingConfig.NUM_EPOCHS,
                        help="Number of epochs to train.", type=int)
    parser.add_argument("--print_every", required=False, default=ReportingConfig.PRINT_EVERY,
                        help="Number of epochs to train.", type=int)    
    parser.add_argument("--learning_rate_disc", required=False, default=TrainingConfig.LEARNING_RATE_DISC,
                        help="Learning rate for the discriminator")
    parser.add_argument("--learning_rate_gen", required=False, default=TrainingConfig.LEARNING_RATE_GEN,
                        help="Learning rate for the generator")
    parser.add_argument("--learning_rate_q", required=False, default=TrainingConfig.LEARNING_RATE_Q,
                        help="Learning rate for the generator")
    parser.add_argument("--latent_dim", required=False, type=int, default=TrainingConfig.LATENT_NOISE_DIM,
                        help="The number of elements of pure noise")
    parser.add_argument("--model_dir", required=False,
                        default=GlobalConfig.get("MODEL_DIR"))
    parser.add_argument("--dataset_name", required=False, type=str,
                        default=TrainingConfig.DATASET_NAME)
    parser.add_argument("--batch_size", required=False, type=int,
                        default=TrainingConfig.BATCH_SIZE)
    parser.add_argument("--num_categories", required=False, type=int,
                        default=TrainingConfig.NUM_CATEGORIES)
    parser.add_argument("--num_continuous", required=False, type=int,
                        default=TrainingConfig.NUM_CONTINUOUS)
    parser.add_argument("--disc_train_interval", required=False, type=int,
                        default=TrainingConfig.DISCRIM_TRAIN_INTERVAL)
    parser.add_argument("--noise_scale", required=False, type=int,
                        default=TrainingConfig.IMAGE_NOISE_SCALE)
    parser.add_argument("--model_in_folder", required=False,
                        help="Path to a folder containing disc / generator weights. The latest one will be loaded.")
    args = parser.parse_args()

    train(args)
