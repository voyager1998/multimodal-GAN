import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from openTSNE import TSNE

# Torch related
import torch
from torch import nn

# Local modules
from datasets import Edge2Shoe
from models import (ResNetGenerator,
                    Encoder,
                    reparameterization)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_latents', action="store_true",
                        help='whether latent variables are already stored')
    parser.add_argument('--load_tsne', action="store_true",
                        help='whether t-SNE results are already stored')

    opt = parser.parse_args()

    if not opt.__dict__['load_latents'] and not opt.__dict__['load_tsne']:
        # Training Configurations
        # (You may put your needed configuration here. Please feel free to add more or use argparse. )
        checkpoints_path = 'checkpoints_archived/'

        img_dir = 'data/edges2shoes/train/'
        img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
        n_residual_blocks = 6
        batch_size = 1
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

        # Define generator, encoder and discriminators
        generator = ResNetGenerator(latent_dim, img_shape, n_residual_blocks, device=gpu_id).to(gpu_id)
        encoder = Encoder(latent_dim).to(gpu_id)

        epoch_id = 19
        path = os.path.join(checkpoints_path, 'bicycleGAN_epoch_' + str(epoch_id))
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        encoder.eval()
        generator.eval()

        all_latents = []
        for idx, data in tqdm(enumerate(loader)):
            # if idx > 10:
            #     break
            # ######## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
            real_A = edge_tensor
            real_B = rgb_tensor

            """
            B -> encoded latent -> B
            """
            z_mu, z_logvar = encoder.forward(rgb_tensor)
            z_encoded = reparameterization(z_mu, z_logvar, device=gpu_id)

            all_latents.append(z_encoded.detach().cpu())

        latents = torch.cat(all_latents, dim=0)
        print(latents.shape)
        np.save("all_latents", latents.detach().cpu().numpy())

    elif not opt.__dict__['load_tsne']:
        latents = np.load("all_latents.npy")

        Y = TSNE().fit(latents)
        np.save("tSNE_result", Y)

    else:
        Y = np.load("tSNE_result.npy")
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], 20)
        plt.savefig("latent_visualization.png")
        plt.show()
