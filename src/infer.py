import argparse
import os
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt

# Torch related
import torch
from torch import nn, optim
from torch.autograd import Variable

# Local modules
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
    checkpoints_path = 'checkpoints/'
    test_img_dir = 'out_images_infer/'
    os.makedirs(test_img_dir, exist_ok=True)

    img_dir = 'data/edges2shoes/val/'
    img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
    n_residual_blocks = 6
    num_epochs = 20
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

    epoch_id = 19
    path = os.path.join(checkpoints_path, 'bicycleGAN_epoch_' + str(epoch_id))
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    encoder.eval()
    generator.eval()
    discriminator.eval()

    # Plot training losses
    list_vae_G_train_loss = checkpoint['list_vae_G_train_loss']
    list_clr_G_train_loss = checkpoint['list_clr_G_train_loss']
    list_kld_train_loss = checkpoint['list_kld_train_loss']
    list_img_train_loss = checkpoint['list_img_train_loss']
    list_G_train_loss = checkpoint['list_G_train_loss']
    list_latent_train_loss = checkpoint['list_latent_train_loss']
    list_vae_D_train_loss = checkpoint['list_vae_D_train_loss']
    list_clr_D_train_loss = checkpoint['list_clr_D_train_loss']

    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(list_vae_G_train_loss)
    axs[0, 1].plot(list_clr_G_train_loss)
    axs[0, 2].plot(list_kld_train_loss)
    axs[1, 0].plot(list_img_train_loss)
    axs[1, 1].plot(list_G_train_loss)
    axs[1, 2].plot(list_latent_train_loss)
    axs[2, 0].plot(list_vae_D_train_loss)
    axs[2, 1].plot(list_clr_D_train_loss)
    axs[0, 0].set_title('list_vae_G_train_loss')
    axs[0, 1].set_title('list_clr_G_train_loss')
    axs[0, 2].set_title('list_kld_train_loss')
    axs[1, 0].set_title('list_img_train_loss')
    axs[1, 1].set_title('list_G_train_loss')
    axs[1, 2].set_title('list_latent_train_loss')
    axs[2, 0].set_title('list_vae_D_train_loss')
    axs[2, 1].set_title('list_clr_D_train_loss')
    plt.show()

    # For adversarial loss (optional to use)
    valid = 1
    fake = 0

    for idx, data in enumerate(loader):
        # ######## Process Inputs ##########
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
        real_A = edge_tensor
        real_B = rgb_tensor

        # -------------------------------
        #  Forward ALL
        # ------------------------------
        fig, axs = plt.subplots(1, 5, figsize=(5, 5))
        vis_real_A = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
        axs[0].imshow(vis_real_A.transpose(1, 2, 0))
        axs[0].set_title('real images')

        z_random = torch.randn(4, real_A.shape[0], latent_dim).to(gpu_id)
        last_fake_B = None
        for i in range(4):
            fake_B_random = generator.forward(real_A, z_random[i])

            # -------------------------------
            #  Visualization
            # ------------------------------
            vis_fake_B_random = denorm(fake_B_random[0].detach()).cpu().data.numpy().astype(np.uint8)
            if last_fake_B is None:
                last_fake_B = vis_fake_B_random
            else:
                isSame = last_fake_B == vis_fake_B_random
                print(np.sum(isSame))
                print(last_fake_B.size)

            axs[i + 1].imshow(vis_fake_B_random.transpose(1, 2, 0))

        path = os.path.join(test_img_dir, 'epoch_' + str(epoch_id) + '_' + str(idx) + '.png')
        plt.savefig(path)
