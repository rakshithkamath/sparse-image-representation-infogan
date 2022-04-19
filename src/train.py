"""
Train a GAN
"""
import pdb
import os
import argparse
from re import T
import time
import glob
from black import TRANSFORMED_MAGICS
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from pathlib import Path
from config import GlobalConfig, TrainingConfig
from models import define_gan, define_discriminator_and_recognition, define_generator
from utils import generate_real_samples, generate_latent_points, generate_fake_samples
from visualization import summarize_performance
# from data import TrainingData

def custom_loss(y, y_pred):
    loss = tf.keras.losses.CategoricalCrossentropy()(y, y_pred)
    pdb.set_trace()
    return loss

def train(args):

    latent_dim = args.latent_dim
    num_cat = 10
    gen_input_size = latent_dim + num_cat

    gen_model = define_generator(gen_input_size)
    disc_model, q_model = define_discriminator_and_recognition(num_cat)
    gan_model = define_gan(gen_model, disc_model, q_model)


    # Make new sub folder for this particular run
    this_time_folder = os.path.join(args.model_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    Path(this_time_folder).mkdir(parents=True, exist_ok=True)

    if args.model_in_folder:
        this_time_folder = args.model_in_folder
        joined_path = os.path.join(args.model_in_folder, "*_generator_model.h5")
        poss_files = glob.glob(joined_path)
        latest_i = max([int(os.path.split(x)[1].split("_")[0]) for x in poss_files])
        gen_model.load_weights(os.path.join(args.model_in_folder, f"{latest_i}_generator_model.h5"))
        gan_model.load_weights(os.path.join(args.model_in_folder, f"{latest_i}_gan_model.h5"))
        disc_model.load_weights(os.path.join(args.model_in_folder, f"{latest_i}_disc_model.h5"))
        offset = latest_i
    else:
        offset = 0

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    dataset = x_train/(127.5) - 1
    
    
    # compile models
    disc_model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=args.learning_rate_disc, beta_1=args.adam_beta))
    opt = optimizers.Adam(lr=args.learning_rate_gen, beta_1=args.adam_beta)
    gan_model.compile(loss=[tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.CategoricalCrossentropy()], optimizer=opt, loss_weights=[1, TrainingConfig.RELATIVE_LOSS])
    
    batch_per_epoch = int(dataset.shape[0] / args.batch_size)
    num_steps = batch_per_epoch * args.epochs
    for iter in range(num_steps):
        i = iter + offset
        # Discriminator Training
        X_real, y_real = generate_real_samples(dataset, int(args.batch_size / 2))
        X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, num_cat, int(args.batch_size / 2))
        d_loss1 = disc_model.train_on_batch(X_real, y_real)
        d_loss2 = disc_model.train_on_batch(X_fake, y_fake)

        # Generator Training
        z_input, cat_codes = generate_latent_points(latent_dim, num_cat, args.batch_size)
        y_gan = np.ones((args.batch_size, 1))


        # print("before predict")
        # gan_output, cat_code_output = gan_model(z_input)

        _, g_1, g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])

        
        # pdb.set_trace()

        print(f"i={i+1}, Disc Loss (real, fake)=({d_loss1:.3f} {d_loss2:.3f}), Gen Loss:{g_1:.3f} Q Loss {g_2:.3f}")

        if (i+1) % (batch_per_epoch * 5) == 0:
            summarize_performance(this_time_folder, i, gen_model, gan_model, latent_dim, num_cat)
            summarize_performance(this_time_folder, i+100000, gen_model, gan_model, latent_dim, num_cat)
            gen_model.save(os.path.join(this_time_folder, f"{(i+1)}_generator_model.h5"))
            gan_model.save(os.path.join(this_time_folder, f"{(i+1)}_gan_model.h5"))
            disc_model.save(os.path.join(this_time_folder, f"{(i+1)}_disc_model.h5"))
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=TrainingConfig.NUM_EPOCHS,
                        help="Number of epochs to train.", type=int)
    parser.add_argument("--learning_rate_disc", required=False, default=TrainingConfig.LEARNING_RATE_DISC,
                        help="Learning rate for the discriminator")
    parser.add_argument("--learning_rate_gen", required=False, default=TrainingConfig.LEARNING_RATE_GEN,
                        help="Learning rate for the generator")
    parser.add_argument("--adam_beta", required=False, default=TrainingConfig.ADAM_BETA,
                        help="Beta")
    parser.add_argument("--latent_dim", required=False, default=TrainingConfig.LATENT_NOISE_DIM,
                        help="The number of elements of pure noise")
    parser.add_argument("--model_dir", required=False,
                        default=GlobalConfig.get("MODEL_DIR"))
    parser.add_argument("--batch_size", required=False, type=int,
                        default=TrainingConfig.BATCH_SIZE)
    parser.add_argument("--model_in_folder", required=False,
        help="Path to a folder containing disc / generator weights. The latest one will be loaded.")
    args = parser.parse_args()

    train(args)