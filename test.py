import torch
from torch import nn

from project.model.ghost_unet import SqueezeExcitation, GhostModule, GhostLayer, GhostUNet, GhostUNetDConv


def main():
    print('Encoder test:')
    x = torch.rand((1, 3, 512, 512))
    gm = GhostUNet()

    y = gm(x)

    print('Decoder test:')
    x = torch.rand((1, 1024, 42, 42))

    convt = nn.ConvTranspose2d(1024, 19, kernel_size=4, stride=2)
    dconv = GhostUNetDConv(19, 19)

    y = dconv(convt(x))
    print(y.shape)
    # TODO: should be: 78x78, is: 86x86


if __name__ == '__main__':
    main()
