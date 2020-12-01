from torch import nn, optim
from vis_tools import visualizer
from datasets import Edge2Shoe
from models import ResNetGenerator, PatchGANDiscriminator, Encoder, weights_init_normal
import argparse
import os
import itertools
import numpy as np
import torch
import time


def norm(image):
    """
    Normalize image tensor
    """
    return (image / 255.0 - 0.5) * 2.0


def denorm(tensor):
    """
    Denormalize image tensor
    """
    return ((tensor + 1.0) / 2.0) * 255.0


if __name__ == "__main__":
    # Training Configurations
    # (You may put your needed configuration here. Please feel free to add more or use argparse. )
    img_dir = 'data/edges2shoes/train/'
    img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
    n_residual_blocks = 6
    num_epochs = 40
    batch_size = 8
    lr_rate = 2e-4	      # Adam optimizer learning rate
    betas = (0.5, 0.999)    # Adam optimizer beta 1, beta 2
    lambda_pixel = 10      # Loss weights for pixel loss
    lambda_latent = 0.5    # Loss weights for latent regression
    lambda_kl = 0.01        # Loss weights for kl divergence
    latent_dim = 8      # latent dimension for the encoded images from domain B
    gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Reparameterization helper function
    # (You may need this helper function here or inside models.py, depending on your encoder
    #   implementation)

    # Random seeds (optional)
    torch.manual_seed(1)
    np.random.seed(1)

    # Define DataLoader
    dataset = Edge2Shoe(img_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Loss functions
    mae_loss = torch.nn.L1Loss().to(gpu_id)

    # Define generator, encoder and discriminators
    generator = ResNetGenerator(latent_dim, img_shape, n_residual_blocks).to(gpu_id)
    encoder = Encoder(latent_dim).to(gpu_id)
    discriminator_VAE = PatchGANDiscriminator(img_shape).to(gpu_id)
    discriminator_LR = PatchGANDiscriminator(img_shape).to(gpu_id)

    # init weights
    generator.apply(weights_init_normal)
    encoder.apply(weights_init_normal)
    discriminator_VAE.apply(weights_init_normal)
    discriminator_LR.apply(weights_init_normal)

    # Define optimizers for networks
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
    optimizer_D_VAE = torch.optim.Adam(discriminator_VAE.parameters(), lr=lr_rate, betas=betas)
    optimizer_D_LR = torch.optim.Adam(discriminator_LR.parameters(), lr=lr_rate, betas=betas)

    # For adversarial loss (optional to use)
    valid = 1
    fake = 0

    # Training
    total_steps = len(loader) * num_epochs
    step = 0
    for e in range(num_epochs):
        start = time.time()
        for idx, data in enumerate(loader):

            # ######## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
            real_A = edge_tensor
            real_B = rgb_tensor

            # -------------------------------
            #  Train Generator and Encoder
            # ------------------------------

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------

            """
            Optional TODO:
                1. You may want to visualize results during training for debugging purpose
                2. Save your model every few iterations
            """
