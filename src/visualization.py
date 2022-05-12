import matplotlib.pyplot as plt
from utils import generate_all_cat_fake_samples, generate_ordered_latent_codes, generate_latent_points
import os
import pdb
import numpy as np

def summarize_performance(output_folder, step, gen_model, gan_model, latent_dim, num_cat, num_samples=9):
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
    X = generate_all_cat_fake_samples(gen_model, latent_dim, num_cat)
    # scale from [-1,1] to [0,1]
    # X = (X + 1) / 2.0

    fig=plt.figure()
    for i in range(num_samples):
        fig.add_subplot(3, 3, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    filename1 = os.path.join(output_folder, f'generated_plot_{step+1}.png')
    fig.tight_layout()  
    fig.savefig(filename1)
    plt.close()


def summarize_performance_tsne(output_folder, step, gen_model, gan_model, latent_dim, num_cat, num_continuous, num_samples=9):
    """

    Args:
        step: Step number
        gen_model: generator model
        gan_model: GAN model
        latent_dim: Latent variables dimension
        num_cat: number of categorical variables
        num_continuous: Number of continuous variables
        num_samples: number of samples that need to be created

    Returns:

    """
    # Generator Training
    latent_code, cat_codes, contin_codes = generate_latent_points(latent_dim, num_cat, num_continuous, num_samples)
    X = gen_model(latent_code, training=True)
    X = X.numpy()


def summarize_performance_continuous(output_folder, step, gen_model, gan_model, latent_dim, num_cat, num_continuous, num_samples=9):
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
    
    for contin_i in range(num_continuous):
        cat_info = [num_cat]
        other_cat_fixed_val = [4]
        target_cat = 0
        num_samples_per_cat = num_cat

        contin_info = [10]*num_continuous
        other_continous_fixed_val = [0]*num_continuous
        target_continuous = contin_i
        num_samples_per_continuous = num_cat

        latent_codes = generate_ordered_latent_codes("span_both",
                                    cat_info, 
                                    other_cat_fixed_val, 
                                    target_cat, 
                                    num_samples_per_cat, 
                                    contin_info, 
                                    other_continous_fixed_val, 
                                    target_continuous, 
                                    num_samples_per_continuous)
        latent_codes = latent_codes.reshape((num_cat*num_samples_per_continuous, -1))
        X = gen_model(latent_codes, training=True)
        X = X.numpy()
        

        # X = generate_all_cat_fake_samples(gen_model, latent_dim, num_cat)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0

        X *= 255
        X = X.astype(np.int32)


        fig=plt.figure(figsize=(100, 100))
        for i in range(num_samples_per_continuous*num_cat):
            fig.add_subplot(num_cat, num_samples_per_continuous, 1 + i)
            plt.axis('off')
            plt.imshow(X[i, :, :, :], cmap='gray_r')
        filename1 = os.path.join(output_folder, f'generated_plot_{step+1}_contin_{contin_i}.png')
        fig.tight_layout()  
        fig.savefig(filename1)
        plt.close()


