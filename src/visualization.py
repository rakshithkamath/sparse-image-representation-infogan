import matplotlib.pyplot as plt
from utils import generate_all_cat_fake_samples, generate_ordered_latent_codes
import os
import pdb

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


def summarize_performance_continuous(output_folder, step, gen_model, gan_model, latent_dim, num_cat, num_samples=9):
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
    
    num_cat = 10
    cat_info = [num_cat]
    other_cat_fixed_val = [4]
    target_cat = 0
    num_samples_per_cat = 5

    contin_info = [2]
    other_continous_fixed_val = [None]
    target_continuous = 0
    num_samples_per_continuous = 5

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

    # X = generate_all_cat_fake_samples(gen_model, latent_dim, num_cat)
    # scale from [-1,1] to [0,1]
    # X = (X + 1) / 2.0

    fig=plt.figure(figsize=(50, 100))
    for i in range(num_samples_per_continuous*num_cat):
        fig.add_subplot(num_cat, num_samples_per_continuous, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    filename1 = os.path.join(output_folder, f'generated_plot_{step+1}.png')
    fig.tight_layout()  
    fig.savefig(filename1)
    plt.close()


