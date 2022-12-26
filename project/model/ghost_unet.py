from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import math


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels, reduction_ratio: float = 16) -> None:
        super().__init__()

        self.channels = input_channels
        excitation_bottleneck_channels = int(self.channels / reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shrink = nn.Linear(self.channels, excitation_bottleneck_channels)
        self.blow = nn.Linear(excitation_bottleneck_channels, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze
        z = self.avg_pool(x).view((-1, self.channels))

        # Excitation
        s = F.relu(self.shrink(z))
        s = torch.sigmoid(self.blow(s)).view((-1, self.channels, 1, 1))

        # Scale
        y = s * x

        return y


class GhostModule(nn.Module):
    # Mostly stolen from https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    # (original authors' implementation)
    # as paper doesn't quite explain what are those cheap operations.
    # Adapted for Ghost-UNet
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 1, ratio: float = 2, dw_size: int = 3, stride: int = 1, relu: bool = True) -> None:
        super().__init__()

        reduced_channels = math.ceil(output_channels / ratio)
        # nearly equal to `output_channels` but multiple
        # of `reduced_channels` for sure, which is required
        # for depth-wise conv
        cheap_channels = reduced_channels * (ratio - 1)

        # as cheap conv may produce a little bit more channels
        # than it is necessary, we store the amount of required channels
        # so as to be able to discard unneeded ones later
        self.output_channels = output_channels

        self.normal_conv = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, kernel_size,
                      stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(reduced_channels, cheap_channels, dw_size, 1,
                      padding=dw_size//2, groups=reduced_channels, bias=False),
            nn.BatchNorm2d(cheap_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.normal_conv(x)
        x2 = self.cheap_conv(x1)

        x = torch.cat([x1, x2], dim=1)
        x = x[:, :self.output_channels, :, :]

        return x


class GhostLayer(nn.Module):
    # NOTE: As there is no implementation and paper is not reproducible at all
    # many sizes here were choosen arbitraily
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, use_se: bool = True) -> None:
        super().__init__()

        self.use_se = use_se
        self.skip_conv = stride == 1

        self.gm1 = GhostModule(in_channels, out_channels)

        # TODO: Figure 1 suggests that when stride = 1 dw conv is skipped, yet on Figure 2 it is always used...
        if not self.skip_conv:
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size,
                                  stride, padding=2, groups=out_channels, bias=False)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        # TODO: Also, sizes do not match. For GL6 input size is 19x19, kernel size 2, stride 1 and output is 20x20
        # similar thing whit layer 7 - size also grows by 1 pixel
        # if dw-conv should really be skipped, some padding must happen inside Ghost Module

        if use_se:
            self.se = SqueezeExcitation(out_channels)

        self.gm2 = GhostModule(out_channels, out_channels, relu=False)

        # TODO: MaxPool how?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gm1(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)

        if not self.skip_conv:
            x = self.conv(x)
            x = self.batch_norm(x)
            x = F.leaky_relu(x)

        if self.use_se:
            x = self.se(x)

        x = self.gm2(x)

        return x


class GhostUNetDConv(nn.Module):
    # At least this is easy
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_features, out_features,
                               kernel_size=3, padding='same')
        self.batch_norm = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)

        return x


class GhostUNet(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.layer1 = GhostLayer(3, 64, 2, 2, False)
        self.layer2 = GhostLayer(64, 125, 2, 2, True)
        self.layer3 = GhostLayer(125, 256, 2, 2, False)
        self.layer4 = GhostLayer(256, 415, 2, 2, True)
        self.layer5 = GhostLayer(415, 612, 2, 2, False)
        self.layer6 = GhostLayer(612, 950, 2, 1, True)
        self.layer7 = GhostLayer(950, 1024, 2, 1, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)

        x = self.layer2(x)
        print(x.shape)

        x = self.layer3(x)
        print(x.shape)

        x = self.layer4(x)
        print(x.shape)

        x = self.layer5(x)
        print(x.shape)

        x = self.layer6(x)
        print(x.shape)

        x = self.layer7(x)
        print(x.shape)

        return x
