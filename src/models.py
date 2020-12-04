from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, latent_dim, input_shape, num_residual_blocks, device='cuda'):
        super(ResNetGenerator, self).__init__()

        self.device = device

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels + latent_dim, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        z = torch.stack([z[i].expand(x.shape[2], x.shape[3], -1).permute(2, 0, 1)
                         for i in range(z.shape[0])]).to(self.device)
        return self.model(torch.cat([x, z], dim=1))


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


def reparameterization(mean, log_var, eps=None, device='cuda'):
    ################################
    # std = exp(0.5*log_var)
    # eps = N(0,1)
    # z = mean + std * eps
    ################################

    std = torch.exp(0.5 * log_var)
    if eps is None:
        eps = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).to(device)
    z = mean + std * eps

    return z


def loss_discriminator(fake_B, D, real_B, Valid_label, Fake_label, criterion):
    '''
    loss_discriminator function is applied to compute loss for discriminator D_A and D_B,
    For example, we want to compute loss for D_A. The loss is consisted of two parts: 
    D(real_A) and D(G(real_B)). We want to penalize the distance between D(real_A) part and 1 and 
    distance between D(G(real_B)) part and 0. 
    We will want to first compute discriminator loss given real_A and valid, which is all 1.
    Then we want to forward real_A through G_AB network to get fake image batch 
    and compute discriminator loss given fake batch and fake, which is all 0.
    Finall, add up these two loss as the total discriminator loss.

    It's important to notice that D(A, B) in the paper is only computing the MSE loss between
    D(B) and Trues.
    '''
    # forward real_B images into the discriminator
    real_out = D.forward(real_B)
    # compute loss between Valid_label and discriminator output on real_B images
    loss_real = criterion(real_out, Valid_label)

    # Compute loss between Fake_label and discriminator output on fake images
    fake_out = D.forward(fake_B.detach())
    loss_fake = criterion(fake_out, Fake_label)
    # sum real_B loss and fake loss as the loss_D
    loss_D = loss_real + loss_fake

    return loss_D


def loss_generator(G, img_A, D, Valid_label, criterion):
    '''
    loss_generator function is applied to compute loss for both generator G_AB and G_BA:
    For example, we want to compute the loss for G_AB.
    img_A will be the real image in domain A, then we map img_A into domain B to get fake B,
    then we compute the loss between D_B(fake_B) and valid, which is all 1.
    The fake_B image will also be one of the outputs, since we want to use it in the loss_cycle_consis.
    '''

    fake_B = G.forward(img_A)
    # forward fake_B images to the discriminator
    fake_out = D.forward(fake_B)

    # Compute loss between valid labels and discriminator output on fake_B images
    loss_G = criterion(fake_out, Valid_label)

    return loss_G, fake_B


def loss_KLD(mu, log_var, device='cuda'):
    '''
    Compute KL divergence loss
    mu, log_var will be the computed from Encoder outputs
    '''
    loss = 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - torch.ones(mu.shape).to(device) - log_var)
    return loss


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
