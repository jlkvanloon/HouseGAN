import torch
import torch.nn as nn

from models.cmp import CMP
from models.conv_block import conv_block


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            *conv_block(9, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"))
        self.l1 = nn.Sequential(nn.Linear(10, 8 * 32 ** 2))
        self.cmp_1 = CMP(in_channels=16)
        self.downsample_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_2 = CMP(in_channels=16)
        self.downsample_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 2, 1, act="leaky"),
            *conv_block(256, 128, 3, 2, 1, act="leaky"),
            *conv_block(128, 128, 3, 2, 1, act="leaky"))

        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))
        self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        x = x.view(-1, 1, 32, 32)
        y = self.l1(given_y)
        y = y.view(-1, 8, 32, 32)
        x = torch.cat([x, y], 1)
        x = self.encoder(x)
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, x.shape[1])

        # global loss
        x_g = add_pool(x, nd_to_sample)
        validity_global = self.fc_layer_global(x_g)

        # local loss
        return validity_global
