import matplotlib.pyplot as plt
from utils import generate_all_cat_fake_samples
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