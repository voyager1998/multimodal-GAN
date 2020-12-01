from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb


##############################
#        Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        """
        The encoder used in both cVAE-GAN and cLR-GAN, which encode img B or B_hat to latent vector
        This encoder uses resnet-18 to extract features, and further encode them into a distribution
        similar to VAE encoder.

        Note: You may either add "reparametrization trick" and "KL divergence" in the train.py file

        Args in constructor:
            latent_dim: latent dimension for z

        Args in forward function:
            img: image input (from domain B)

        Returns:
            mu: mean of the latent code
            logvar: sigma of the latent code
        """
        super(Encoder, self).__init__()

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator
##############################
class Generator(nn.Module):
    """
    The generator used in both cVAE-GAN and cLR-GAN, which transform A to B

    Args in constructor:
        latent_dim: latent dimension for z
        image_shape: (channel, h, w), you may need this to specify the output dimension (optional)

    Args in forward function:
        x: image input (from domain A)
        z: latent vector (encoded B)

    Returns:
            fake_B: generated image in domain B
    """

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        # (TODO: add layers...)

    def forward(self, x, z):
        # (TODO: add layers...)

        return


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        """
        The discriminator used in both cVAE-GAN and cLR-GAN

        Args in constructor:
            in_channels: number of channel in image (default: 3 for RGB)

        Args in forward function:
            x: image input (real_B, fake_B)

        Returns:
            discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        super(Discriminator, self).__init__()

    def forward(self, x):

        return


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(PatchGANDiscriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


from datasets import Edge2Shoe
if __name__ == "__main__":
    # Define DataLoader
    img_dir = 'data/edges2shoes/train/'
    img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose

    # load dataset
    dataset = Edge2Shoe(img_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    net_discriminator = PatchGANDiscriminator(input_shape=img_shape)

    for data in loader:
        edge_tensor, rgb_tensor = data

        discriminator_out = net_discriminator.forward(edge_tensor)
        print(discriminator_out.shape, net_discriminator.output_shape)
