import argparse
import os
import itertools
import numpy as np
import time

# Torch related
import torch
from torch import nn, optim
from torch.autograd import Variable

# Local modules
from vis_tools import visualizer
from datasets import Edge2Shoe
from models import (ResNetGenerator, PatchGANDiscriminator,
                    Encoder, weights_init_normal,
                    reparameterization, loss_KLD,
                    loss_discriminator, loss_generator
                    )


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
    l1_loss = torch.nn.L1Loss().to(gpu_id)
    mse_loss = torch.nn.MSELoss().to(gpu_id)

    # Define generator, encoder and discriminators
    generator = ResNetGenerator(latent_dim, img_shape, n_residual_blocks, device=gpu_id).to(gpu_id)
    encoder = Encoder(latent_dim).to(gpu_id)
    discriminator = PatchGANDiscriminator(img_shape).to(gpu_id)

    # init weights
    generator.apply(weights_init_normal)
    encoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Define optimizers for networks
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=betas)

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

            # Adversarial ground truths
            valid = Variable(torch.Tensor(np.ones((real_A.size(0), *discriminator.output_shape))),
                             requires_grad=False).to(gpu_id)
            fake = Variable(torch.Tensor(np.zeros((real_A.size(0), *discriminator.output_shape))),
                            requires_grad=False).to(gpu_id)

            # -------------------------------
            #  Forward ALL
            # ------------------------------
            encoder.train()
            generator.train()

            z_mu, z_logvar = encoder.forward(rgb_tensor)
            z_encoded = reparameterization(z_mu, z_logvar, device=gpu_id)

            fake_B_encoded = generator.forward(real_A, z_encoded)

            z_random = torch.randn(batch_size, latent_dim)
            fake_B_random = generator.forward(real_A, z_random)

            z_mu_predict, z_logvar_predict = encoder.forward(fake_B_random)

            # -------------------------------
            #  Train Generator and Encoder
            # ------------------------------
            for param in discriminator.parameters():
                param.requires_grad = False

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # G(A) should fool D
            fake_B_encoded_label = discriminator.forward(fake_B_encoded)
            vae_G_loss = mse_loss(fake_B_encoded_label, valid)
            fake_B_random_label = discriminator.forward(fake_B_random)
            clr_G_loss = mse_loss(fake_B_random_label, valid)

            # compute KLD loss
            kld_loss = loss_KLD(z_mu, z_logvar, device=gpu_id)

            # Compute L1 image loss
            img_loss = l1_loss(fake_B_encoded, real_B)

            loss_G = vae_G_loss + clr_G_loss + kld_loss + img_loss
            loss_G.backward(retain_graph=True)
            optimizer_E.step()
            optimizer_G.step()

            # -------------------------------
            #  Train Discriminator
            # ------------------------------
            for param in discriminator.parameters():
                param.requires_grad = True

            optimizer_D.zero_grad()

            # Compute VAE-GAN disciminator loss
            vae_D_loss = loss_discriminator(fake_B_encoded, discriminator, real_B, valid, fake, mse_loss)
            vae_D_loss.backward()

            clr_D_loss = loss_discriminator(fake_B_random, discriminator, real_B, valid, fake, mse_loss)
            clr_D_loss.backward()

            optimizer_D.step()

            print(loss_G, vae_D_loss, clr_D_loss)

            """
            Optional TODO:
                1. You may want to visualize results during training for debugging purpose
                2. Save your model every few iterations
            """
