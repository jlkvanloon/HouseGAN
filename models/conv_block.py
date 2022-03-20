import torch
from torch.nn.utils.parametrizations import spectral_norm


def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False):
    block = []

    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels,
                                                                kernel_size=k, stride=s,
                                                                padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels,
                                                  kernel_size=k, stride=s,
                                                  padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels,
                                                       kernel_size=k, stride=s,
                                                       padding=p, bias=True)))
        else:
            block.append(torch.nn.Conv2d(in_channels, out_channels,

                                         kernel_size=k, stride=s,
                                         padding=p, bias=True))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block
